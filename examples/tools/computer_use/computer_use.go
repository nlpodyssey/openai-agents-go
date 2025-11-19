// Copyright 2025 The NLP Odyssey Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/nlpodyssey/openai-agents-go/agents"
	"github.com/nlpodyssey/openai-agents-go/computer"
	"github.com/nlpodyssey/openai-agents-go/modelsettings"
	"github.com/nlpodyssey/openai-agents-go/tracing"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/playwright-community/playwright-go"
)

func main() {
	// Uncomment to see very verbose logs
	//agents.EnableVerboseStdoutLogging()

	err := WithLocalPlaywrightComputer(func(comp *LocalPlaywrightComputer) error {
		return tracing.RunTrace(
			context.Background(), tracing.TraceParams{WorkflowName: "Computer use example"},
			func(ctx context.Context, _ tracing.Trace) error {
				agent := agents.New("Browser user").
					WithInstructions("You are a helpful agent.").
					WithTools(agents.ComputerTool{Computer: comp}).
					// Use the computer using model, and set truncation to auto because its required
					WithModel("computer-use-preview").
					WithModelSettings(modelsettings.ModelSettings{
						Truncation: param.NewOpt(modelsettings.TruncationAuto),
					})

				result, err := agents.Run(ctx, agent, "Search for SF sports news and summarize.")
				if err != nil {
					return err
				}

				fmt.Println(result.FinalOutput)
				return nil
			},
		)
	})
	if err != nil {
		panic(err)
	}
}

var cuaKeyToPlaywrightKey = map[string]string{
	"/":          "Divide",
	"\\":         "Backslash",
	"alt":        "Alt",
	"arrowdown":  "ArrowDown",
	"arrowleft":  "ArrowLeft",
	"arrowright": "ArrowRight",
	"arrowup":    "ArrowUp",
	"backspace":  "Backspace",
	"capslock":   "CapsLock",
	"cmd":        "Meta",
	"ctrl":       "Control",
	"delete":     "Delete",
	"end":        "End",
	"enter":      "Enter",
	"esc":        "Escape",
	"home":       "Home",
	"insert":     "Insert",
	"option":     "Alt",
	"pagedown":   "PageDown",
	"pageup":     "PageUp",
	"shift":      "Shift",
	"space":      " ",
	"super":      "Meta",
	"tab":        "Tab",
	"win":        "Meta",
}

// LocalPlaywrightComputer is a computer implemented using a local Playwright browser.
type LocalPlaywrightComputer struct {
	dimensions computer.Dimensions
	pw         *playwright.Playwright
	browser    playwright.Browser
	page       playwright.Page
}

func WithLocalPlaywrightComputer(fn func(*LocalPlaywrightComputer) error) (err error) {
	const width = 1024
	const height = 768

	pw, err := playwright.Run()
	if err != nil {
		return fmt.Errorf("error starting playwright: %w", err)
	}
	defer func() {
		if e := pw.Stop(); e != nil {
			err = errors.Join(err, fmt.Errorf("error stopping playwright: %w", e))
		}
	}()

	headless := false
	browser, err := pw.Chromium.Launch(playwright.BrowserTypeLaunchOptions{
		Args: []string{
			fmt.Sprintf("--window-size={%d},{%d}", width, height),
		},
		Headless: &headless,
	})
	if err != nil {
		return fmt.Errorf("error launching browser: %w", err)
	}
	defer func() {
		if e := browser.Close(); e != nil {
			err = errors.Join(err, fmt.Errorf("error closing browser: %w", e))
		}
	}()

	page, err := browser.NewPage()
	if err != nil {
		return fmt.Errorf("error creating page: %w", err)
	}

	err = page.SetViewportSize(width, height)
	if err != nil {
		return fmt.Errorf("error setting viewport size: %w", err)
	}

	if _, err = page.Goto("https://www.bing.com"); err != nil {
		return fmt.Errorf("error navigating to URL: %w", err)
	}

	return fn(&LocalPlaywrightComputer{
		dimensions: computer.Dimensions{Width: width, Height: height},
		pw:         pw,
		browser:    browser,
		page:       page,
	})
}

func (c *LocalPlaywrightComputer) Environment(context.Context) (computer.Environment, error) {
	return computer.EnvironmentBrowser, nil
}

func (c *LocalPlaywrightComputer) Dimensions(context.Context) (computer.Dimensions, error) {
	return c.dimensions, nil
}

func (c *LocalPlaywrightComputer) Screenshot(context.Context) (string, error) {
	// Capture only the viewport (not full_page).
	fullPage := false
	pngBytes, err := c.page.Screenshot(playwright.PageScreenshotOptions{
		FullPage: &fullPage,
	})
	if err != nil {
		return "", fmt.Errorf("screenshot error: %w", err)
	}
	return base64.StdEncoding.EncodeToString(pngBytes), nil
}

func (c *LocalPlaywrightComputer) Click(_ context.Context, x, y int64, button computer.Button) error {
	// Playwright only supports left, middle, right buttons
	var pwButton *playwright.MouseButton
	switch button {
	case computer.ButtonLeft:
		pwButton = playwright.MouseButtonLeft
	case computer.ButtonRight:
		pwButton = playwright.MouseButtonRight
	case computer.ButtonWheel:
		pwButton = playwright.MouseButtonMiddle
	default:
		pwButton = playwright.MouseButtonLeft
	}

	err := c.page.Mouse().Click(float64(x), float64(y), playwright.MouseClickOptions{Button: pwButton})
	if err != nil {
		return fmt.Errorf("click error: %w", err)
	}
	return nil
}

func (c *LocalPlaywrightComputer) DoubleClick(_ context.Context, x, y int64) error {
	err := c.page.Mouse().Dblclick(float64(x), float64(y))
	if err != nil {
		return fmt.Errorf("double click error: %w", err)
	}
	return nil
}

func (c *LocalPlaywrightComputer) Scroll(_ context.Context, x, y int64, scrollX, scrollY int64) error {
	err := c.page.Mouse().Move(float64(x), float64(y))
	if err != nil {
		return fmt.Errorf("mouse move to scroll error: %w", err)
	}
	_, err = c.page.Evaluate(fmt.Sprintf("window.scrollBy(%d, %d)", scrollX, scrollY))
	if err != nil {
		return fmt.Errorf("page scroll error: %w", err)
	}
	return nil
}

func (c *LocalPlaywrightComputer) Type(_ context.Context, text string) error {
	err := c.page.Keyboard().Type(text)
	if err != nil {
		return fmt.Errorf("keyboard typing error: %w", err)
	}
	return nil
}

func (c *LocalPlaywrightComputer) Wait(context.Context) error {
	time.Sleep(1 * time.Second)
	return nil
}

func (c *LocalPlaywrightComputer) Move(_ context.Context, x, y int64) error {
	err := c.page.Mouse().Move(float64(x), float64(y))
	if err != nil {
		return fmt.Errorf("mouse move error: %w", err)
	}
	return nil

}

func (c *LocalPlaywrightComputer) Keypress(_ context.Context, keys []string) error {
	mappedKeys := make([]string, len(keys))
	for i, key := range keys {
		if v, ok := cuaKeyToPlaywrightKey[strings.ToLower(key)]; ok {
			mappedKeys[i] = v
		} else {
			mappedKeys[i] = key
		}
	}

	for _, key := range mappedKeys {
		err := c.page.Keyboard().Down(key)
		if err != nil {
			return fmt.Errorf("key down error: %w", err)
		}
	}

	for _, key := range mappedKeys {
		err := c.page.Keyboard().Up(key)
		if err != nil {
			return fmt.Errorf("key up error: %w", err)
		}
	}

	return nil
}

func (c *LocalPlaywrightComputer) Drag(_ context.Context, path []computer.Position) error {
	if len(path) == 0 {
		return nil
	}
	err := c.page.Mouse().Move(float64(path[0].X), float64(path[0].Y))
	if err != nil {
		return fmt.Errorf("mouse move to drag error: %w", err)
	}
	err = c.page.Mouse().Down()
	if err != nil {
		return fmt.Errorf("mouse down to drag error: %w", err)
	}
	for _, p := range path[1:] {
		err = c.page.Mouse().Move(float64(p.X), float64(p.Y))
		if err != nil {
			return fmt.Errorf("mouse move to drag error: %w", err)
		}
	}
	err = c.page.Mouse().Up()
	if err != nil {
		return fmt.Errorf("mouse up to drag error: %w", err)
	}
	return nil
}
