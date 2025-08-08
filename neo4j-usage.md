# Neo4j Setup and Usage

## Quick Start with Docker

1. **Start Neo4j:**
   ```bash
   docker compose -f docker-compose.neo4j.yml up -d
   ```

2. **Check Neo4j status:**
   ```bash
   docker compose -f docker-compose.neo4j.yml ps
   ```

3. **Access Neo4j Browser:**
   - Open: http://localhost:7474
   - Username: `neo4j`
   - Password: `password123`

4. **Stop Neo4j:**
   ```bash
   docker compose -f docker-compose.neo4j.yml down
   ```

## Using with RKG CLI

### Basic commands with Neo4j backend:

```bash
# Analyze schema using Neo4j
rkg analyze -c "postgresql://user:pass@localhost:5432/db" \
    --backend neo4j \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password123

# Create clusters with Neo4j
rkg create-clusters -c "postgresql://user:pass@localhost:5432/db" \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password123

# Get main cluster
rkg get-main-cluster

# Suggest joins with Neo4j
rkg suggest-joins -c "postgresql://user:pass@localhost:5432/db" \
    -t "users,orders" \
    --backend neo4j \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password123

# Schema summary with Neo4j
rkg summary -c "postgresql://user:pass@localhost:5432/db" \
    --backend neo4j \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password password123
```

### Environment Variables (Alternative)

Set these environment variables to avoid passing Neo4j credentials each time:

```bash
export RKG_NEO4J_URI=bolt://localhost:7687
export RKG_NEO4J_USER=neo4j
export RKG_NEO4J_PASSWORD=password123
```

## Connection Details

- **Bolt Protocol:** `bolt://localhost:7687`
- **HTTP:** `http://localhost:7474`
- **Username:** `neo4j`
- **Password:** `password123`

## Troubleshooting

1. **Connection refused:**
   - Check if container is running: `docker ps`
   - Check logs: `docker compose -f docker-compose.neo4j.yml logs neo4j`

2. **Memory issues:**
   - Adjust heap sizes in docker-compose.yml
   - Default: 512m initial, 1G max

3. **Data persistence:**
   - Data is stored in Docker volume `neo4j_data`
   - To reset: `docker compose -f docker-compose.neo4j.yml down -v`

## Performance Comparison

| Backend | Setup | Memory | Speed (<1k tables) | Speed (>1k tables) |
|---------|-------|--------|-------------------|-------------------|
| NetworkX | Simple | High | Fast | Slow |
| Neo4j | Docker | Low | Medium | Fast |