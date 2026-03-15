#!/usr/bin/env bash
set -e

BRANCH="claude/install-google-stitch-mcp-zt9ob"
RAW="https://raw.githubusercontent.com/eljefe06/propengine-xz5982/$BRANCH"

# ── 1. Detectar webroot de Apache ────────────────────────────────
echo "==> Detectando webroot de Apache..."

WEBROOT=""

# Buscar en los vhosts activos de Apache
if command -v apachectl &>/dev/null; then
  WEBROOT=$(apachectl -S 2>/dev/null \
    | grep -i "DocumentRoot" \
    | head -1 \
    | grep -oE '/[^"[:space:]]+')
fi

# Fallback: revisar rutas comunes
if [ -z "$WEBROOT" ]; then
  for candidate in \
    /var/www/myrock.com.mx \
    /var/www/myrock \
    /var/www/html \
    /home/*/public_html \
    /srv/www/myrock; do
    if [ -d "$candidate" ]; then
      WEBROOT="$candidate"
      break
    fi
  done
fi

if [ -z "$WEBROOT" ]; then
  echo "❌ No se pudo detectar el webroot. Especifícalo manualmente:"
  echo "   WEBROOT=/ruta/correcta bash deploy-myrock.sh"
  exit 1
fi

echo "    Webroot encontrado: $WEBROOT"

# ── 2. Descargar archivos ─────────────────────────────────────────
echo ""
echo "==> Descargando archivos desde GitHub..."

curl -fsSL "$RAW/myrock/index.html"                -o "$WEBROOT/index.html"    && echo "    ✓ index.html"
curl -fsSL "$RAW/myrock/style.css"                 -o "$WEBROOT/style.css"     && echo "    ✓ style.css"
curl -fsSL "$RAW/myrock/producto.html"             -o "$WEBROOT/producto.html" && echo "    ✓ producto.html"
curl -fsSL "$RAW/myrock/producto.css"              -o "$WEBROOT/producto.css"  && echo "    ✓ producto.css"
curl -fsSL "$RAW/manual-myrock-mail-engine.html"   -o "$WEBROOT/manual-myrock-mail-engine.html" && echo "    ✓ manual-myrock-mail-engine.html"

# ── 3. Permisos ───────────────────────────────────────────────────
chown -R www-data:www-data "$WEBROOT" 2>/dev/null || true
chmod -R 755 "$WEBROOT"
echo "    ✓ Permisos aplicados"

# ── 4. Recargar servidor web ──────────────────────────────────────
echo ""
echo "==> Recargando servidor web..."

if systemctl is-active --quiet apache2; then
  systemctl reload apache2
  echo "    ✓ Apache recargado"
elif systemctl is-active --quiet httpd; then
  systemctl reload httpd
  echo "    ✓ HTTPD recargado"
elif systemctl is-active --quiet nginx; then
  nginx -t && systemctl reload nginx
  echo "    ✓ Nginx recargado"
else
  echo "    ⚠ No se detectó servidor web activo. Recarga manualmente si es necesario."
fi

# ── 5. Listo ──────────────────────────────────────────────────────
echo ""
echo "✅ Deploy completado"
echo "   Archivos en: $WEBROOT"
echo "   Sitio:       https://myrock.com.mx"
echo "   Producto:    https://myrock.com.mx/producto.html"
echo "   Manual:      https://myrock.com.mx/manual-myrock-mail-engine.html"
