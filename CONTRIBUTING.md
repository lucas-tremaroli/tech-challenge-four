# Contributing

This document provides instructions on how to work with the condebase.

## Running InfluxDB 3 with Docker

To run InfluxDB 3 using Docker, you can use the following command:

```bash
docker compose up --build -d
```

To generate a new admin token, use the following command:

```bash
docker exec -it influxdb3-container influxdb3 create token --admin
```
