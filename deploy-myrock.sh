#!/usr/bin/env bash
set -e

BRANCH="claude/install-google-stitch-mcp-zt9ob"
RAW="https://raw.githubusercontent.com/eljefe06/propengine-xz5982/$BRANCH"

# Permite override: WEBROOT=/otra/ruta bash deploy-myrock.sh
WEBROOT="${WEBROOT:-/var/www/myrock}"

echo "==> Desplegando MyRock en $WEBROOT"
mkdir -p "$WEBROOT"

echo ""
echo "==> Descargando archivos..."
curl -fsSL "$RAW/myrock/index.html"              -o "$WEBROOT/index.html"                    && echo "    ✓ index.html"
curl -fsSL "$RAW/myrock/style.css"               -o "$WEBROOT/style.css"                     && echo "    ✓ style.css"
curl -fsSL "$RAW/myrock/producto.html"           -o "$WEBROOT/producto.html"                 && echo "    ✓ producto.html"
curl -fsSL "$RAW/myrock/producto.css"            -o "$WEBROOT/producto.css"                  && echo "    ✓ producto.css"
curl -fsSL "$RAW/manual-myrock-mail-engine.html" -o "$WEBROOT/manual-myrock-mail-engine.html" && echo "    ✓ manual-myrock-mail-engine.html"

echo ""
echo "==> Aplicando permisos..."
chown -R www-data:www-data "$WEBROOT" 2>/dev/null || true
chmod -R 755 "$WEBROOT"
echo "    ✓ Listo"

echo ""
echo "==> Recargando servidor web..."
if systemctl is-active --quiet apache2; then
  systemctl reload apache2 && echo "    ✓ Apache recargado"
elif systemctl is-active --quiet httpd; then
  systemctl reload httpd && echo "    ✓ HTTPD recargado"
elif systemctl is-active --quiet nginx; then
  nginx -t && systemctl reload nginx && echo "    ✓ Nginx recargado"
else
  echo "    ⚠ Recarga el servidor manualmente si es necesario."
fi

echo ""
echo "✅ Deploy completado"
echo "   https://myrock.com.mx"
echo "   https://myrock.com.mx/producto.html"
echo "   https://myrock.com.mx/manual-myrock-mail-engine.html"
