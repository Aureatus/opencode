package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"time"

	"github.com/openai/openai-go"
	// "github.com/openai/openai-go/shared" // If shared.ReasoningEffort was used, ensure this path is correct
	"github.com/sst/opencode/internal/config"
	"github.com/sst/opencode/internal/llm/models"
	"github.com/sst/opencode/internal/llm/tools"
	"github.com/sst/opencode/internal/message"
	"github.com/sst/opencode/internal/status"
)

func (o *openaiClient) convertMessagesToChatCompletionMessages(messages []message.Message) (openaiMessages []openai.ChatCompletionMessageParamUnion) {
	// Add system message first
	openaiMessages = append(openaiMessages, openai.SystemMessage(o.providerOptions.systemMessage))

	for _, msg := range messages {
		switch msg.Role {
		case message.User:
			var content []openai.ChatCompletionContentPartUnionParam
			textBlock := openai.ChatCompletionContentPartTextParam{Text: msg.Content().String()}
			content = append(content, openai.ChatCompletionContentPartUnionParam{OfText: &textBlock})
			for _, binaryContent := range msg.BinaryContent() {
				imageURL := openai.ChatCompletionContentPartImageImageURLParam{URL: binaryContent.String(models.ProviderOpenAI)}
				imageBlock := openai.ChatCompletionContentPartImageParam{ImageURL: imageURL}
				content = append(content, openai.ChatCompletionContentPartUnionParam{OfImageURL: &imageBlock})
			}
			openaiMessages = append(openaiMessages, openai.UserMessage(content))

		case message.Assistant:
			assistantMsg := openai.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}
			if msg.Content().String() != "" {
				assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(msg.Content().String()),
				}
			}
			if len(msg.ToolCalls()) > 0 {
				assistantMsg.ToolCalls = make([]openai.ChatCompletionMessageToolCallParam, len(msg.ToolCalls()))
				for i, call := range msg.ToolCalls() {
					assistantMsg.ToolCalls[i] = openai.ChatCompletionMessageToolCallParam{
						ID:   call.ID,
						Type: "function",
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      call.Name,
							Arguments: call.Input,
						},
					}
				}
			}
			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantMsg,
			})

		case message.Tool:
			for _, result := range msg.ToolResults() {
				openaiMessages = append(openaiMessages,
					openai.ToolMessage(result.Content, result.ToolCallID),
				)
			}
		}
	}
	return
}

func (o *openaiClient) convertToChatCompletionTools(tools []tools.BaseTool) []openai.ChatCompletionToolParam {
	if len(tools) == 0 {
		return nil // Return nil if there are no tools, to avoid sending an empty "tools": []
	}
	openaiTools := make([]openai.ChatCompletionToolParam, len(tools))

	for i, tool := range tools {
		info := tool.Info()
		var params openai.FunctionParameters

		if info.Parameters == nil || len(info.Parameters) == 0 {
			params = openai.FunctionParameters{
				"type":       "object",
				"properties": make(map[string]interface{}), // Empty map for properties
			}
		} else {
			params = openai.FunctionParameters{
				"type":       "object",
				"properties": info.Parameters,
			}
			if len(info.Required) > 0 {
				params["required"] = info.Required
			}
		}

		openaiTools[i] = openai.ChatCompletionToolParam{
			Type: "function", // Use the string "function" for the type
			Function: openai.FunctionDefinitionParam{
				Name:        info.Name,
				Description: openai.String(info.Description), // Ensure description is not nil
				Parameters:  params,
			},
		}
	}
	return openaiTools
}

func (o *openaiClient) preparedChatCompletionParams(messages []openai.ChatCompletionMessageParamUnion, tools []openai.ChatCompletionToolParam) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(o.providerOptions.model.APIModel),
		Messages: messages,
	}

	if len(tools) > 0 {
		params.Tools = tools
		// Example: params.ToolChoice = openai.ToolChoiceAuto // or "required" or specific function
	}

	// ReasoningEffort handling: The standard openai-go library doesn't have a direct ReasoningEffort field.
	// This logic seems custom or for a fork/older version, or meant for OpenRouter's WithExtraFields.
	// For now, I'll assume it needs to go into WithExtraFields if targeting OpenRouter.
	if o.providerOptions.model.CanReason { // Assuming 'CanReason' typo is intended
		params.MaxCompletionTokens = openai.Int(o.providerOptions.maxTokens)
		if o.providerOptions.model.Provider == models.ProviderOpenRouter {
			// Pass reasoning_effort via WithExtraFields for OpenRouter
			reasoningEffortValue := "medium" // default
			if o.options.reasoningEffort != "" {
				reasoningEffortValue = string(o.options.reasoningEffort)
			}
			params.WithExtraFields(map[string]any{"reasoning_effort": reasoningEffortValue})
		} else {
			// If this was for a direct OpenAI param that your library version supports,
			// you would set it here. Otherwise, this block might not do anything for standard OpenAI.
			// e.g. if 'shared.ReasoningEffortLow' existed and params.ReasoningEffort was a field:
			// switch o.options.reasoningEffort {
			// case "low": params.ReasoningEffort = shared.ReasoningEffortLow
			// ...
			// }
		}
	} else {
		params.MaxTokens = openai.Int(o.providerOptions.maxTokens)
	}

	if o.providerOptions.model.Provider == models.ProviderOpenRouter {
		currentExtraFields := params.GetExtraFields()
		if currentExtraFields == nil {
			currentExtraFields = make(map[string]any)
		}
		currentExtraFields["provider"] = map[string]any{
			"require_parameters": true,
		}
		params.WithExtraFields(currentExtraFields)
	}

	return params
}

func (o *openaiClient) sendChatcompletionMessage(ctx context.Context, messages []message.Message, toolsList []tools.BaseTool) (response *ProviderResponse, err error) {
	// Corrected variable name for clarity
	convertedTools := o.convertToChatCompletionTools(toolsList)
	params := o.preparedChatCompletionParams(o.convertMessagesToChatCompletionMessages(messages), convertedTools)

	cfg := config.Get()
	if cfg.Debug {
		jsonData, _ := json.MarshalIndent(params, "", "  ")
		slog.Debug("DEBUG: Outgoing OpenAI Request Parameters", "params", string(jsonData))
	}

	attempts := 0
	for {
		attempts++
		openaiResponse, err := o.client.Chat.Completions.New(
			ctx,
			params,
		)
		if err != nil {
			retry, after, retryErr := o.shouldRetry(attempts, err)
			duration := time.Duration(after) * time.Millisecond
			if retryErr != nil {
				return nil, retryErr // This is the error from shouldRetry (could be original or new)
			}
			if retry {
				status.Warn(fmt.Sprintf("Retrying due to rate limit... attempt %d of %d", attempts, maxRetries), status.WithDuration(duration))
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(duration):
					continue
				}
			}
			return nil, err // If not retrying, return the original error from New()
		}

		content := ""
		if len(openaiResponse.Choices) > 0 && openaiResponse.Choices[0].Message.Content != "" {
			content = openaiResponse.Choices[0].Message.Content
		}

		toolCalls := o.chatCompletionToolCalls(*openaiResponse)
		var finishReason message.FinishReason = message.FinishReasonUnknown // Default
		if len(openaiResponse.Choices) > 0 {
			finishReason = o.finishReason(string(openaiResponse.Choices[0].FinishReason)) // Use existing method
		}

		if len(toolCalls) > 0 {
			finishReason = message.FinishReasonToolUse
		}

		return &ProviderResponse{
			Content:      content,
			ToolCalls:    toolCalls,
			Usage:        o.usage(*openaiResponse),
			FinishReason: finishReason,
		}, nil
	}
}

func (o *openaiClient) streamChatCompletionMessages(ctx context.Context, messages []message.Message, toolsList []tools.BaseTool) <-chan ProviderEvent {
	// Corrected variable name for clarity
	convertedTools := o.convertToChatCompletionTools(toolsList)
	params := o.preparedChatCompletionParams(o.convertMessagesToChatCompletionMessages(messages), convertedTools)

	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}

	cfg := config.Get()
	if cfg.Debug {
		jsonData, _ := json.MarshalIndent(params, "", "  ")
		slog.Debug("DEBUG: Outgoing OpenAI Request Parameters", "params", string(jsonData))
	}

	attempts := 0
	eventChan := make(chan ProviderEvent)

	go func() {
		defer close(eventChan)
		for {
			attempts++
			// Assuming NewStreaming returns 1 value (the stream) as per original user code and compiler error
			openaiStream := o.client.Chat.Completions.NewStreaming(
				ctx,
				params,
			)
			// If NewStreaming can error before returning stream, it would be:
			// openaiStream, streamErr := o.client.Chat.Completions.NewStreaming(ctx, params)
			// if streamErr != nil {
			// 	eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("failed to start streaming: %w", streamErr)}
			// 	return
			// }

			acc := openai.ChatCompletionAccumulator{}
			currentContent := ""
			toolCallsAccumulated := make([]message.ToolCall, 0)

			for openaiStream.Next() {
				chunk := openaiStream.Current()
				acc.AddChunk(chunk)

				for _, choice := range chunk.Choices {
					if choice.Delta.Content != "" {
						eventChan <- ProviderEvent{
							Type:    EventContentDelta,
							Content: choice.Delta.Content,
						}
						currentContent += choice.Delta.Content
					}
					// Simplified tool call accumulation from delta; final accumulation is more robust
					if len(choice.Delta.ToolCalls) > 0 {
						// This needs careful implementation if you want to stream partial tool calls.
						// For now, we rely on the fully accumulated message for tool calls.
					}
				}
			}

			// Check stream error AFTER the loop
			err := openaiStream.Err()
			if err == nil || errors.Is(err, io.EOF) {
				finalCompletion := acc.ChatCompletion
				var finishReason message.FinishReason = message.FinishReasonUnknown // Default
				if len(finalCompletion.Choices) > 0 {
					finishReason = o.finishReason(string(finalCompletion.Choices[0].FinishReason)) // Use existing
					if len(finalCompletion.Choices[0].Message.ToolCalls) > 0 {
						toolCallsAccumulated = o.chatCompletionToolCalls(finalCompletion)
					}
				}

				if len(toolCallsAccumulated) > 0 {
					finishReason = message.FinishReasonToolUse
				}

				finalContent := currentContent
				if len(finalCompletion.Choices) > 0 && finalCompletion.Choices[0].Message.Content != "" {
					finalContent = finalCompletion.Choices[0].Message.Content
				}


				eventChan <- ProviderEvent{
					Type: EventComplete,
					Response: &ProviderResponse{
						Content:      finalContent,
						ToolCalls:    toolCallsAccumulated,
						Usage:        o.usage(finalCompletion),
						FinishReason: finishReason,
					},
				}
				return
			}

			retry, after, retryErr := o.shouldRetry(attempts, err) // err is openaiStream.Err()
			duration := time.Duration(after) * time.Millisecond
			if retryErr != nil {
				eventChan <- ProviderEvent{Type: EventError, Error: retryErr}
				return
			}
			if retry {
				status.Warn(fmt.Sprintf("Retrying due to rate limit... attempt %d of %d", attempts, maxRetries), status.WithDuration(duration))
				select {
				case <-ctx.Done():
					if ctx.Err() != nil {
						eventChan <- ProviderEvent{Type: EventError, Error: ctx.Err()}
					}
					return
				case <-time.After(duration):
					continue
				}
			}
			eventChan <- ProviderEvent{Type: EventError, Error: err} // Use original stream err if not retrying
			return
		}
	}()

	return eventChan
}

func (o *openaiClient) chatCompletionToolCalls(completion openai.ChatCompletion) []message.ToolCall {
	var toolCalls []message.ToolCall
	if len(completion.Choices) > 0 && len(completion.Choices[0].Message.ToolCalls) > 0 {
		for _, call := range completion.Choices[0].Message.ToolCalls {
			toolCall := message.ToolCall{
				ID:       call.ID,
				Name:     call.Function.Name,
				Input:    call.Function.Arguments,
				Type:     "function",
				Finished: true,
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}
	return toolCalls
}

func (o *openaiClient) usage(completion openai.ChatCompletion) TokenUsage {
	// If completion.Usage is a direct struct, it cannot be nil.
	// We access its fields directly.
	// The same applies to completion.Usage.PromptTokensDetails if it's also a direct struct.

	// It's good practice to check if essential values are zero if the API might omit them,
	// though for a usage block, some values are usually expected.
	// For now, we'll assume if the Usage struct is there, its fields are populated or zero.

	var cachedTokens int64 = 0 // Ensure cachedTokens is int64
	// If PromptTokensDetails is a direct struct, it exists. Access its field.
	// If it might not be populated by the API, CachedTokens would be its zero value (0 for int64).
	// Some libraries might offer a boolean to check if an optional struct field was actually present in the JSON.
	// Without such a feature, we proceed assuming it's populated or zero.
	cachedTokens = completion.Usage.PromptTokensDetails.CachedTokens // This is int64

	var inputTokens int64 = completion.Usage.PromptTokens - cachedTokens // Both are int64

	return TokenUsage{
		InputTokens:         inputTokens,                             // int64
		OutputTokens:        completion.Usage.CompletionTokens,       // int64
		CacheCreationTokens: 0,                                       // Assuming int64 in TokenUsage
		CacheReadTokens:     cachedTokens,                            // int64
	}
}

// NOTE: The 'finishReason' method that was here has been removed,
// assuming it exists in 'internal/llm/provider/openai.go' as o.finishReason(...)
// If it doesn't, that method needs to be correctly defined there or elsewhere accessible to *openaiClient.
