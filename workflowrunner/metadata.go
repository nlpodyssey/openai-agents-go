package workflowrunner

import (
	"strings"
)

func composeTraceMetadata(req WorkflowRequest) map[string]any {
	metadata := map[string]any{
		"session_id":    req.Session.SessionID,
		"user_id":       req.Session.Credentials.UserID,
		"account_id":    req.Session.Credentials.AccountID,
		"workflow_name": req.Workflow.Name,
	}
	if len(req.Session.Credentials.Capabilities) > 0 {
		metadata["capabilities"] = strings.Join(req.Session.Credentials.Capabilities, ",")
	}
	for k, v := range req.Metadata {
		if _, reserved := metadata[k]; reserved {
			continue
		}
		metadata[k] = v
	}
	return metadata
}
