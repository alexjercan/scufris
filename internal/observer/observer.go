package observer

import (
	"context"
)

type Observer interface {
	OnStart(ctx context.Context)
	OnToken(ctx context.Context, token string) error
	OnEnd(ctx context.Context)

	OnError(ctx context.Context, err error)

	OnImage(ctx context.Context, imageId string) error
	OnToolCall(ctx context.Context, toolName string, args any) error
	OnToolCallEnd(ctx context.Context, toolName string, result any) error
}

type Multi struct {
	observers []Observer
}

func NewMulti(obs ...Observer) Observer {
	return &Multi{observers: obs}
}

func (m *Multi) OnStart(ctx context.Context) {
	for _, obs := range m.observers {
		obs.OnStart(ctx)
	}
}

func (m *Multi) OnToken(ctx context.Context, token string) error {
	for _, obs := range m.observers {
		if err := obs.OnToken(ctx, token); err != nil {
			return err
		}
	}

	return nil
}

func (m *Multi) OnEnd(ctx context.Context) {
	for _, obs := range m.observers {
		obs.OnEnd(ctx)
	}
}

func (m *Multi) OnError(ctx context.Context, err error) {
	for _, obs := range m.observers {
		obs.OnError(ctx, err)
	}
}

func (m *Multi) OnImage(ctx context.Context, imageId string) error {
	for _, obs := range m.observers {
		if err := obs.OnImage(ctx, imageId); err != nil {
			return err
		}
	}

	return nil
}

func (m *Multi) OnToolCall(ctx context.Context, toolName string, args any) error {
	for _, obs := range m.observers {
		if err := obs.OnToolCall(ctx, toolName, args); err != nil {
			return err
		}
	}

	return nil
}

func (m *Multi) OnToolCallEnd(ctx context.Context, toolName string, result any) error {
	for _, obs := range m.observers {
		if err := obs.OnToolCallEnd(ctx, toolName, result); err != nil {
			return err
		}
	}

	return nil
}

type observerKeyType struct{}

var observerKey = observerKeyType{}

func WithObserver(ctx context.Context, observer Observer) context.Context {
	return context.WithValue(ctx, observerKey, observer)
}

func GetObserver(ctx context.Context) (Observer, bool) {
	name, ok := ctx.Value(observerKey).(Observer)
	return name, ok
}

func OnStart(ctx context.Context) {
	if obs, ok := GetObserver(ctx); ok {
		obs.OnStart(ctx)
	}
}

func OnToken(ctx context.Context, token string) error {
	if obs, ok := GetObserver(ctx); ok {
		return obs.OnToken(ctx, token)
	}
	return nil
}

func OnEnd(ctx context.Context) {
	if obs, ok := GetObserver(ctx); ok {
		obs.OnEnd(ctx)
	}
}

func OnError(ctx context.Context, err error) {
	if obs, ok := GetObserver(ctx); ok {
		obs.OnError(ctx, err)
	}
}

func OnImage(ctx context.Context, imageId string) error {
	if obs, ok := GetObserver(ctx); ok {
		return obs.OnImage(ctx, imageId)
	}
	return nil
}

func OnToolCall(ctx context.Context, toolName string, args any) error {
	if obs, ok := GetObserver(ctx); ok {
		return obs.OnToolCall(ctx, toolName, args)
	}
	return nil
}

func OnToolCallEnd(ctx context.Context, toolName string, result any) error {
	if obs, ok := GetObserver(ctx); ok {
		return obs.OnToolCallEnd(ctx, toolName, result)
	}
	return nil
}
