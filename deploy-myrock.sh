#!/usr/bin/env bash
set -e

BRANCH="claude/install-google-stitch-mcp-zt9ob"
REPO="https://github.com/eljefe06/propengine-xz5982"

# WordPress vive dentro del contenedor Docker myrock-wordpress
# El volumen montado en /var/www/html está en el host aquí:
WEBROOT="${WEBROOT:-/var/lib/docker/volumes/myrock-stack_wordpress_data/_data}"
PLUGINS_DIR="$WEBROOT/wp-content/plugins"

echo "==> Desplegando MyRock"
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

# ─── Plugin: eliminar duplicado ──────────────────────────────────────────────
echo ""
echo "==> Limpiando plugins duplicados..."
if [ -d "$PLUGINS_DIR/myrock-mail-engine-1" ]; then
  rm -rf "$PLUGINS_DIR/myrock-mail-engine-1"
  echo "    ✓ Eliminado myrock-mail-engine-1"
else
  echo "    — myrock-mail-engine-1 no existe, nada que eliminar"
fi

# ─── Plugin: desplegar versión definitiva ────────────────────────────────────
echo ""
echo "==> Desplegando plugin myrock-mail-engine..."
rm -rf "$PLUGINS_DIR/myrock-mail-engine"
cp -r "$TMPDIR/repo/myrock-mail-engine" "$PLUGINS_DIR/myrock-mail-engine"
echo "    ✓ Plugin desplegado"

# ─── Permisos ────────────────────────────────────────────────────────────────
echo ""
echo "==> Aplicando permisos..."
chown -R www-data:www-data "$WEBROOT" 2>/dev/null || true
chmod -R 755 "$WEBROOT"
echo "    ✓ Listo"

echo ""
echo "✅ Deploy completado"
echo "   https://myrock.com.mx"
echo "   https://myrock.com.mx/producto.html"
echo "   https://myrock.com.mx/manual-myrock-mail-engine.html"
