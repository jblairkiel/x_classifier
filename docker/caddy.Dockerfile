# syntax=docker/dockerfile:1.7
FROM caddy:2.10.2-alpine@sha256:4c6e91c6ed0e2fa03efd5b44747b625fec79bc9cd06ac5235a779726618e530d

# Run unprivileged on 8080/8443 (host 80/443 are mapped to them), so the container needs no
# capabilities at all. Copying the binary drops its file capability (cap_net_bind_service),
# which would otherwise make exec fail under cap_drop: ALL.
RUN cp /usr/bin/caddy /tmp/caddy && mv /tmp/caddy /usr/bin/caddy \
 && addgroup -g 1002 -S caddy && adduser -u 1002 -S -G caddy -H -h /nonexistent caddy \
 && mkdir -p /data /config && chown -R caddy:caddy /data /config

COPY docker/Caddyfile /etc/caddy/Caddyfile
USER caddy
EXPOSE 8080 8443
