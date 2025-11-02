# Contributing

This document provides instructions on how to work with the condebase.

## Running InfluxDB 3 with Docker

To run InfluxDB 3 using Docker, you can use the following command:

```bash
docker run -it -p 8181:8181 --name influxdb3-container \
      --volume ~/.influxdb3_data:/.data --volume ~/.influxdb3_plugins:/plugins influxdb:3-core \
      influxdb3 serve --node-id node0 --object-store file --data-dir /.data --plugin-dir /plugins
```

To generate a new admin token, use the following command:

```bash
docker exec -it influxdb3-container influxdb3 create token --admin
```
