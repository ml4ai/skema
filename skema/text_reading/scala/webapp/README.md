# webapp

To create the docker image, use `dockerizeWebapp` from the main project of `sbt`.  The result should be a docker image `skema-webapp`.  Then use `docker-compose up` to run the image with a configuration file similar to this:

`docker-compose.yml`

```
version: '3.2'
services:
  skema-webapp:
    image: skema-webapp:1.0.0
    restart: unless-stopped
    ports:
      - 9005:9000
    container_name: skema-webapp
    environment:
      secret: <secret>
      SKEMA_HOSTNAME: skema.clulab.org
      _JAVA_OPTIONS: -Xmx16g -Xms16g -Dfile.encoding=UTF-8
```

Alternatively, start the container manually with a long command like this:

```
docker run --restart unless-stopped -p 9005:9000 --name skema-webapp \
--env secret=<secret> --env SKEMA_HOSTNAME=skema.clulab.org \
--env _JAVA_OPTIONS="-Xmx16g -Xms16g -Dfile.encoding=UTF-8" \
skema-webapp:1.0.0
```

Do not forget to set the `<secret>`.  Values for the SKEMA_HOSTNAME and the external port (9005) may also need to be adjusted.
