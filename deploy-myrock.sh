#!/usr/bin/env bash
set -e

BRANCH="claude/install-google-stitch-mcp-zt9ob"
REPO="https://github.com/eljefe06/propengine-xz5982"

# WordPress vive dentro del contenedor Docker myrock-wordpress
# El volumen montado en /var/www/html está en el host aquí:
WEBROOT="${WEBROOT:-/var/lib/docker/volumes/myrock-stack_wordpress_data/_data}"

echo "==> Desplegando MyRock"
echo "    WEBROOT: $WEBROOT"

# ─── Clonar repo en directorio temporal ──────────────────────────────────────
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo ""
echo "==> Clonando repositorio (rama $BRANCH)..."
git clone --depth=1 --branch "$BRANCH" "$REPO" "$TMPDIR/repo" --quiet
echo "    ✓ Repo clonado"

# ─── CSS compartido (referenciado desde páginas WP) ──────────────────────────
echo ""
echo "==> Desplegando CSS..."
cp "$TMPDIR/repo/myrock/style.css"    "$WEBROOT/style.css"    && echo "    ✓ style.css"
cp "$TMPDIR/repo/myrock/producto.css" "$WEBROOT/producto.css" && echo "    ✓ producto.css"

# ─── Eliminar HTML estáticos — WordPress maneja el contenido ahora ────────────
echo ""
echo "==> Limpiando archivos HTML estáticos (WordPress toma el control)..."
rm -f "$WEBROOT/index.html"                    && echo "    ✓ index.html eliminado"
rm -f "$WEBROOT/producto.html"                 && echo "    ✓ producto.html eliminado"
rm -f "$WEBROOT/manual-myrock-mail-engine.html" && echo "    ✓ manual-myrock-mail-engine.html eliminado"

# ─── Permisos ────────────────────────────────────────────────────────────────
echo ""
echo "==> Aplicando permisos..."
chown -R www-data:www-data "$WEBROOT/style.css" "$WEBROOT/producto.css" 2>/dev/null || true
echo "    ✓ Listo"

echo ""
echo "✅ Deploy completado — WordPress controla el contenido"
echo "   https://myrock.com.mx"
echo "   https://myrock.com.mx/myrock-mail-engine/"
echo "   https://myrock.com.mx/manual-mail-engine/"
