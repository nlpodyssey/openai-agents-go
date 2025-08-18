#!/bin/bash

# Test script for LiteLLM Docker Integration Example
set -e

echo "🧪 LiteLLM Docker Integration Test Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY not set. Setting a dummy key for testing...${NC}"
    export OPENAI_API_KEY="sk-dummy-key-for-testing"
fi

echo -e "${GREEN}✅ Docker is running${NC}"
echo -e "${GREEN}✅ API key is set${NC}"

# Check if LiteLLM container is already running
if docker ps | grep -q "litellm"; then
    echo -e "${GREEN}✅ LiteLLM container is already running${NC}"
else
    echo -e "${YELLOW}📦 Starting LiteLLM Docker container...${NC}"
    
    # Start LiteLLM container
    docker run -d \
        --name litellm-test \
        -v $(pwd)/litellm_config.yaml:/app/config.yaml \
        -e OPENAI_API_KEY="$OPENAI_API_KEY" \
        -p 4000:4000 \
        ghcr.io/berriai/litellm:main-stable \
        --config /app/config.yaml \
        --detailed_debug
    
    echo -e "${GREEN}✅ LiteLLM container started${NC}"
    
    # Wait for container to be ready
    echo -e "${YELLOW}⏳ Waiting for LiteLLM to be ready...${NC}"
    sleep 10
fi

# Test LiteLLM endpoint
echo -e "${YELLOW}🔍 Testing LiteLLM endpoint...${NC}"
response=$(curl -s -o /dev/null -w "%{http_code}" \
    --location 'http://localhost:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
        "model": "openai-gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1
    }')

if [ "$response" = "200" ]; then
    echo -e "${GREEN}✅ LiteLLM endpoint is responsive${NC}"
else
    echo -e "${RED}❌ LiteLLM endpoint returned HTTP $response${NC}"
    echo -e "${YELLOW}📋 Container logs:${NC}"
    docker logs litellm-test --tail 20
    exit 1
fi

# Run the Go example
echo -e "${YELLOW}🚀 Running Go example...${NC}"
if go run main.go; then
    echo -e "${GREEN}✅ Go example completed successfully!${NC}"
else
    echo -e "${RED}❌ Go example failed${NC}"
    exit 1
fi

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    docker stop litellm-test > /dev/null 2>&1 || true
    docker rm litellm-test > /dev/null 2>&1 || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Ask if user wants to keep container running
echo ""
read -p "Keep LiteLLM container running? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    cleanup
else
    echo -e "${GREEN}🐳 LiteLLM container will keep running on port 4000${NC}"
    echo -e "${YELLOW}To stop it later, run: docker stop litellm-test && docker rm litellm-test${NC}"
fi

echo -e "${GREEN}🎉 Test completed!${NC}" 