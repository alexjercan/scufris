package observer

import (
	"context"
)

type TokenObserver interface {
	OnStart(ctx context.Context)
	OnToken(ctx context.Context, token string) error
	OnEnd(ctx context.Context)

	OnError(ctx context.Context, err error)

	OnImage(ctx context.Context, imageId string) error
	OnToolCall(ctx context.Context, toolName string, args any) error
	OnToolCallEnd(ctx context.Context, toolName string, result any) error
}

type multiObserver struct {
	observers []TokenObserver
}

func NewMultiObserver(obs ...TokenObserver) TokenObserver {
	return &multiObserver{observers: obs}
}

func (m *multiObserver) OnStart(ctx context.Context) {
	for _, obs := range m.observers {
		obs.OnStart(ctx)
	}
}

func (m *multiObserver) OnToken(ctx context.Context, token string) error {
	for _, obs := range m.observers {
		if err := obs.OnToken(ctx, token); err != nil {
			return err
		}
	}

	return nil
}

func (m *multiObserver) OnEnd(ctx context.Context) {
	for _, obs := range m.observers {
		obs.OnEnd(ctx)
	}
}

func (m *multiObserver) OnError(ctx context.Context, err error) {
	for _, obs := range m.observers {
		obs.OnError(ctx, err)
	}
}

func (m *multiObserver) OnImage(ctx context.Context, imageId string) error {
	for _, obs := range m.observers {
		if err := obs.OnImage(ctx, imageId); err != nil {
			return err
		}
	}

	return nil
}

func (m *multiObserver) OnToolCall(ctx context.Context, toolName string, args any) error {
	for _, obs := range m.observers {
		if err := obs.OnToolCall(ctx, toolName, args); err != nil {
			return err
		}
	}

	return nil
}

func (m *multiObserver) OnToolCallEnd(ctx context.Context, toolName string, result any) error {
	for _, obs := range m.observers {
		if err := obs.OnToolCallEnd(ctx, toolName, result); err != nil {
			return err
		}
	}

	return nil
}

type observerKeyType struct{}

var observerKey = observerKeyType{}

func WithObserver(ctx context.Context, observer TokenObserver) context.Context {
	return context.WithValue(ctx, observerKey, observer)
}

func GetObserver(ctx context.Context) (TokenObserver, bool) {
	name, ok := ctx.Value(observerKey).(TokenObserver)
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
