#!/usr/bin/env bash
set -e

BRANCH="claude/install-google-stitch-mcp-zt9ob"
REPO="https://github.com/eljefe06/propengine-xz5982"

# WordPress vive dentro del contenedor Docker myrock-wordpress
# El volumen montado en /var/www/html está en el host aquí:
WEBROOT="${WEBROOT:-/var/lib/docker/volumes/myrock-stack_wordpress_data/_data}"

echo "==> Desplegando MyRock (archivos estáticos)"
echo "    WEBROOT: $WEBROOT"

# ─── Clonar repo en directorio temporal ──────────────────────────────────────
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo ""
echo "==> Clonando repositorio (rama $BRANCH)..."
git clone --depth=1 --branch "$BRANCH" "$REPO" "$TMPDIR/repo" --quiet
echo "    ✓ Repo clonado"

# ─── Archivos estáticos en raíz de WordPress ─────────────────────────────────
echo ""
echo "==> Desplegando archivos estáticos..."
cp "$TMPDIR/repo/myrock/index.html"              "$WEBROOT/index.html"              && echo "    ✓ index.html"
cp "$TMPDIR/repo/myrock/style.css"               "$WEBROOT/style.css"               && echo "    ✓ style.css"
cp "$TMPDIR/repo/myrock/producto.html"           "$WEBROOT/producto.html"           && echo "    ✓ producto.html"
cp "$TMPDIR/repo/myrock/producto.css"            "$WEBROOT/producto.css"            && echo "    ✓ producto.css"
cp "$TMPDIR/repo/manual-myrock-mail-engine.html" "$WEBROOT/manual-myrock-mail-engine.html" && echo "    ✓ manual-myrock-mail-engine.html"

# ─── Permisos ────────────────────────────────────────────────────────────────
echo ""
echo "==> Aplicando permisos..."
chown -R www-data:www-data "$WEBROOT/index.html" "$WEBROOT/style.css" \
  "$WEBROOT/producto.html" "$WEBROOT/producto.css" \
  "$WEBROOT/manual-myrock-mail-engine.html" 2>/dev/null || true
echo "    ✓ Listo"

echo ""
echo "✅ Deploy completado"
echo "   https://myrock.com.mx"
echo "   https://myrock.com.mx/producto.html"
echo "   https://myrock.com.mx/manual-myrock-mail-engine.html"
