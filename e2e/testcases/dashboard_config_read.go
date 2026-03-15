package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-config-read", pkgtestcases.TestCase{
		Description: "Verify dashboard config endpoints return the router config as JSON and YAML",
		Tags:        []string{"dashboard", "config"},
		Fn:          testDashboardConfigRead,
	})
}

func testDashboardConfigRead(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	httpClient := &http.Client{Timeout: 15 * time.Second}

	// GET /api/router/config/all → must be valid JSON with at least one key.
	jsonURL := fmt.Sprintf("http://localhost:%s/api/router/config/all", localPort)
	if opts.Verbose {
		fmt.Printf("[Dashboard] GET %s\n", jsonURL)
	}

	jsonReq, err := http.NewRequestWithContext(ctx, http.MethodGet, jsonURL, nil)
	if err != nil {
		return fmt.Errorf("create JSON config request: %w", err)
	}

	jsonResp, err := httpClient.Do(jsonReq)
	if err != nil {
		return fmt.Errorf("config/all request failed: %w", err)
	}
	defer jsonResp.Body.Close()

	jsonBody, _ := io.ReadAll(jsonResp.Body)

	if jsonResp.StatusCode != http.StatusOK {
		return fmt.Errorf("config/all: expected 200, got %d: %s", jsonResp.StatusCode, truncateString(string(jsonBody), 200))
	}

	var configJSON map[string]interface{}
	if err := json.Unmarshal(jsonBody, &configJSON); err != nil {
		return fmt.Errorf("config/all response is not valid JSON: %w", err)
	}

	if len(configJSON) == 0 {
		return fmt.Errorf("config/all returned empty JSON object")
	}

	// GET /api/router/config/yaml → must be non-empty with yaml Content-Type.
	yamlURL := fmt.Sprintf("http://localhost:%s/api/router/config/yaml", localPort)
	if opts.Verbose {
		fmt.Printf("[Dashboard] GET %s\n", yamlURL)
	}

	yamlReq, err := http.NewRequestWithContext(ctx, http.MethodGet, yamlURL, nil)
	if err != nil {
		return fmt.Errorf("create YAML config request: %w", err)
	}

	yamlResp, err := httpClient.Do(yamlReq)
	if err != nil {
		return fmt.Errorf("config/yaml request failed: %w", err)
	}
	defer yamlResp.Body.Close()

	yamlBody, _ := io.ReadAll(yamlResp.Body)

	if yamlResp.StatusCode != http.StatusOK {
		return fmt.Errorf("config/yaml: expected 200, got %d: %s", yamlResp.StatusCode, truncateString(string(yamlBody), 200))
	}

	contentType := yamlResp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "yaml") {
		return fmt.Errorf("config/yaml: expected Content-Type to contain 'yaml', got %q", contentType)
	}

	if len(strings.TrimSpace(string(yamlBody))) == 0 {
		return fmt.Errorf("config/yaml returned empty body")
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] config-read OK: JSON keys=%d, YAML bytes=%d\n", len(configJSON), len(yamlBody))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"config_json_keys": len(configJSON),
			"config_yaml_size": len(yamlBody),
		})
	}

	return nil
}
