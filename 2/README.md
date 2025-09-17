# n8n Database Agent with LLM (OpenAI & Ollama)

This project sets up an n8n workflow with a PostgreSQL database that creates an intelligent agent capable of understanding natural language queries and executing appropriate database operations. It includes two implementations:
- OpenAI-based agent (requires API key)
- Ollama-based agent with Mistral 7B (runs locally)

## Prerequisites

- Podman installed on your system
- OpenAI API key (only for OpenAI version, not required for Ollama)

## Setup

### 1. Start the Infrastructure

Run the setup script to create and start both PostgreSQL and n8n containers:

```bash
./setup-podman.sh
```

This will:
- Create a Podman network for container communication
- Start PostgreSQL with sample data (products, customers, orders)
- Start Ollama with Mistral 7B model
- Start n8n with database connectivity

### 2. Access n8n

Once the setup is complete:
- Open your browser and go to: http://localhost:5678
- Login with credentials: `admin` / `admin`

### 3. Configure Database Connection

In n8n, you need to create PostgreSQL credentials:

1. Go to Credentials → Add Credential → PostgreSQL
2. Use these settings:
   - Host: `n8n-postgres`
   - Database: `n8n`
   - User: `n8n`
   - Password: `n8n_password`
   - Port: `5432`
3. Save the credentials

### 4. Import and Configure the Workflow

You have two workflow options:

#### Option A: Using Ollama with AI Agent and SQL Tool (Recommended)

1. In n8n, click on "Workflows" → "Add workflow" → "Import from file"
2. Upload the `n8n-workflow-agent-sql-tool.json` file
3. Update the credentials in the workflow nodes:
   - SQL Tool: Select the PostgreSQL credentials you created
4. Save and activate the workflow

This workflow uses:
- **AI Agent node**: Orchestrates the conversation and tool usage
- **SQL Tool**: PostgreSQL as a tool (not a node) that the agent can use
- **Ollama Chat Model**: Local LLM using Mistral 7B

#### Option B: Using Ollama with HTTP Requests (Alternative)

1. Import the `n8n-workflow-agent-ollama.json` file
2. Update PostgreSQL credentials
3. Save and activate the workflow

This workflow uses HTTP Request nodes to communicate with Ollama.

#### Option C: Using OpenAI (Requires API Key)

1. Go to Credentials → Add Credential → OpenAI
2. Add your OpenAI API key
3. Save the credentials
4. Import the `n8n-workflow-agent.json` file
5. Update the credentials in the workflow nodes:
   - PostgreSQL node: Select the PostgreSQL credentials you created
   - OpenAI nodes: Select the OpenAI credentials you created
6. Save and activate the workflow

## Using the Database Agent

Once the workflow is active, you can send POST requests to the webhook endpoint:

### For Ollama version:
```bash
curl -X POST http://localhost:5678/webhook-test/ai-agent-sql \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all products that cost more than 100 dollars"}'
```

Example queries you can try:
- "Show me all products in the Electronics category"
- "How many customers do we have from Czech Republic?"
- "What is the total value of all pending orders?"
- "List the top 5 most expensive products"
- "Show me all orders with their customer names"

## Database Schema

The database contains three tables:

### Products
- `id`: Primary key
- `name`: Product name
- `description`: Product description
- `price`: Product price
- `category`: Product category
- `stock_quantity`: Available stock

### Customers
- `id`: Primary key
- `first_name`: Customer's first name
- `last_name`: Customer's last name
- `email`: Customer's email
- `phone`: Phone number
- `city`: City
- `country`: Country

### Orders
- `id`: Primary key
- `customer_id`: Foreign key to customers
- `order_date`: When the order was placed
- `total_amount`: Order total
- `status`: Order status (pending, processing, shipped, delivered)
- `shipping_address`: Delivery address

## Managing the Environment

### View Logs
```bash
# View n8n logs
./logs-podman.sh n8n

# View PostgreSQL logs
./logs-podman.sh postgres

# View Ollama logs
./logs-podman.sh ollama
```

### Stop Everything
```bash
./stop-podman.sh
```

### Complete Cleanup
To remove all data and start fresh:
```bash
./stop-podman.sh
podman volume rm postgres_data n8n_data ollama_data
podman network rm n8n-network
```

## Workflow Features

Both n8n workflows include:
1. **Webhook trigger**: Receives HTTP requests with natural language queries
2. **LLM integration**:
   - OpenAI version: Uses GPT-3.5-turbo API
   - Ollama version: Uses local Mistral 7B model
3. **Database tools**: Executes queries against PostgreSQL
4. **Response formatting**: Formats database results in human-readable form
5. **Error handling**: Gracefully handles queries that don't need execution

## Performance Comparison

- **OpenAI**: Faster responses, requires API key, costs per request
- **Ollama**: Slower initial responses (especially first run), completely free, runs locally, more private

## Extending the Agent

You can enhance the agent by:
1. Adding more tables to the database
2. Creating additional tools (e.g., data visualization, export capabilities)
3. Implementing more complex business logic
4. Adding authentication and authorization
5. Integrating with other services (email, Slack, etc.)

## Testing

Test scripts are provided for both implementations:
```bash
# Test Ollama version
./test-agent-ollama.sh

# Test OpenAI version (original)
./test-agent.sh
```

## Troubleshooting

If containers fail to start:
1. Check if ports 5432, 5678, and 11434 are already in use
2. Ensure Podman is running and properly configured
3. Check logs using the provided log scripts
4. Verify network connectivity between containers

For Ollama specific issues:
- First model pull may take time (Mistral 7B is ~4GB)
- Initial responses are slower as the model loads into memory
- Check Ollama logs: `./logs-podman.sh ollama`

## Security Notes

- Change default passwords before production use
- Configure proper authentication for n8n
- Use environment variables for sensitive data
- Implement proper access controls for the database
