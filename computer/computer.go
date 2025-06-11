package computer

import "context"

// A Computer interface abstracts the operations needed to control a computer or browser.
type Computer interface {
	Environment(context.Context) (Environment, error)
	Dimensions(context.Context) (Dimensions, error)
	Screenshot(context.Context) (string, error)
	Click(ctx context.Context, x, y int64, button Button) error
	DoubleClick(ctx context.Context, x, y int64) error
	Scroll(ctx context.Context, x, y int64, scrollX, scrollY int64) error
	Type(ctx context.Context, text string) error
	Wait(context.Context) error
	Move(ctx context.Context, x, y int64) error
	Keypress(ctx context.Context, keys []string) error
	Drag(ctx context.Context, path []Position) error
}

type Dimensions struct {
	Width  int64
	Height int64
}

type Position struct {
	X int64
	Y int64
}

type Environment string

const (
	EnvironmentWindows Environment = "windows"
	EnvironmentMac     Environment = "mac"
	EnvironmentLinux   Environment = "linux"
	EnvironmentUbuntu  Environment = "ubuntu"
	EnvironmentBrowser Environment = "browser"
)

type Button string

const (
	ButtonLeft    Button = "left"
	ButtonRight   Button = "right"
	ButtonWheel   Button = "wheel"
	ButtonBack    Button = "back"
	ButtonForward Button = "forward"
)
