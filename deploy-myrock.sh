#!/usr/bin/env bash
set -e

REPO="https://github.com/eljefe06/propengine-xz5982"
BRANCH="claude/install-google-stitch-mcp-zt9ob"
WEBROOT="/var/www/myrock"
NGINX_CONF="/etc/nginx/sites-available/myrock"

echo "==> Desplegando MyRock en $WEBROOT"

# 1. Descargar archivos directamente del repo
mkdir -p "$WEBROOT"

RAW="https://raw.githubusercontent.com/eljefe06/propengine-xz5982/$BRANCH"

echo "    Descargando index.html..."
curl -fsSL "$RAW/myrock/index.html" -o "$WEBROOT/index.html"

echo "    Descargando style.css..."
curl -fsSL "$RAW/myrock/style.css" -o "$WEBROOT/style.css"

echo "    Estableciendo permisos..."
chown -R www-data:www-data "$WEBROOT"
chmod -R 755 "$WEBROOT"

# 2. Crear virtualhost nginx para myrock.com.mx
cat > "$NGINX_CONF" <<'NGINX'
server {
    listen 80;
    listen [::]:80;

    server_name myrock.com.mx www.myrock.com.mx;

    root /var/www/myrock;
    index index.html;

    # Gzip
    gzip on;
    gzip_types text/css application/javascript text/html;
    gzip_comp_level 6;

    location / {
        try_files $uri $uri/ =404;
    }

    # Cache estáticos
    location ~* \.(css|js|woff2|png|jpg|ico|svg)$ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
NGINX

# 3. Activar el sitio
if [ ! -L /etc/nginx/sites-enabled/myrock ]; then
    ln -s "$NGINX_CONF" /etc/nginx/sites-enabled/myrock
    echo "    Sitio activado en nginx"
fi

# 4. Validar y recargar nginx
nginx -t && systemctl reload nginx

echo ""
echo "✅ DONE — MyRock desplegado correctamente"
echo "   URL: http://31.97.40.155 (o http://myrock.com.mx si el DNS apunta aqui)"
echo ""
echo "   Para agregar SSL (Let's Encrypt):"
echo "   apt install certbot python3-certbot-nginx -y"
echo "   certbot --nginx -d myrock.com.mx -d www.myrock.com.mx"
