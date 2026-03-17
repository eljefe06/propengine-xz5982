# LESSONS.md — Lecciones aprendidas · MyRock Mail Engine

Registro de errores, bugs, decisiones técnicas y soluciones encontradas
durante el desarrollo. Cada entrada incluye contexto para que otro agente
no repita el mismo error.

---

## Plugin: MyRock Mail Engine

### L-01 · SendLog API — métodos correctos

**Contexto:** `CampaignService::send_batch()` llamaba a `SendLog::create()` y
`SendLog::update()` que no existían en el modelo.

**Correcto:**
```php
SendLog::bulk_create( $campaign_id, $recipients ); // crear logs en lote
SendLog::update_status( $log_id, 'sent' );          // actualizar estado
SendLog::find_by_campaign( $campaign_id, ['status'=>'pending','limit'=>50] );
SendLog::count_by_campaign( $campaign_id, 'pending' );
```

**Nunca usar:** `SendLog::create()`, `SendLog::update()` — no existen.

---

### L-02 · MailMessage — API fluida, no array

**Contexto:** Primeras versiones intentaban pasar arrays al `MailManager::send()`.

**Correcto:**
```php
$message = ( new MailMessage() )
    ->setTo( $email, $name )
    ->setSubject( $subject )
    ->setHtml( $html )
    ->setText( $text )
    ->setFrom( $from_email, $from_name )
    ->setReplyTo( $reply_to )
    ->addHeader( 'List-Unsubscribe: <' . $url . '>' );

MailManager::send( $message );
```

---

### L-03 · dbDelta requiere formato exacto

**Contexto:** `dbDelta()` es sensible al formato SQL. Si el SQL no tiene dos
espacios antes de las claves (`KEY`) o falta la línea en blanco correcta,
crea tablas incompletas sin error visible.

**Regla:** Siempre seguir el formato de `Schema.php` existente. No "optimizar"
el SQL de `dbDelta`. Probar activación en entorno limpio tras cualquier cambio.

---

### L-04 · Autoloader PSR-4 — namespaces estrictos

**Contexto:** Clases fuera del namespace `MyRock\MailEngine\` no se cargan
aunque el archivo exista.

**Regla de mapeo:**
```
Namespace: MyRock\MailEngine\Services\CampaignService
Archivo:   includes/Services/CampaignService.php

Namespace: MyRock\MailEngine\Admin\Pages\CampaignsPage
Archivo:   includes/Admin/Pages/CampaignsPage.php
```

El autoloader vive en el entry point `myrock-mail-engine.php`. No tiene
dependencias externas de Composer.

---

### L-05 · WP Cron no corre en servidores sin visitas

**Contexto:** En producción con poco tráfico, `mrme_process_campaigns` puede
retrasarse porque WP Cron se dispara con visitas HTTP.

**Solución recomendada (pendiente de implementar):**
Agregar al crontab del servidor:
```cron
*/5 * * * * wget -q -O - https://myrock.com.mx/wp-cron.php?doing_wp_cron > /dev/null 2>&1
```
Y en `wp-config.php`: `define('DISABLE_WP_CRON', true);`

---

### L-06 · REST API — autenticación doble

El endpoint `mrme/v1/*` acepta dos formas de autenticación:
1. Usuario WordPress logueado con `manage_options` (cookie nonce).
2. Header `X-MRME-Key` con el valor guardado en la opción `mrme_api_key`.

Al hacer pruebas con `curl` usar Application Passwords de WordPress
(formato `xxxx xxxx xxxx xxxx xxxx xxxx`) con `-u "admin:APP_PASSWORD"`.

---

### L-07 · Contact::query() devuelve shape variable

**Contexto:** `Contact::query()` puede devolver `['items'=>[…], 'total'=>N]`
o `['data'=>[…], 'total'=>N]` dependiendo de la versión.

**Defensa:**
```php
$result   = Contact::query( $args );
$contacts = $result['items'] ?? $result['data'] ?? [];
$total    = (int) ( $result['total'] ?? count( $contacts ) );
```

---

### L-08 · Páginas WordPress editadas vía REST, no archivos locales

**Contexto:** El contenido de la homepage (ID 6) vive en la BD de WordPress.
No existe un archivo `.html` local que lo represente.

**Para leer/escribir:**
```python
import requests

# Leer
resp = requests.get(
    "https://myrock.com.mx/wp-json/wp/v2/pages/6?context=edit",
    auth=("admin", "APP_PASSWORD")
)
content = resp.json()['content']['raw']

# Escribir
requests.post(
    "https://myrock.com.mx/wp-json/wp/v2/pages/6",
    json={"content": nuevo_content},
    auth=("admin", "APP_PASSWORD")
)
```

Siempre hacer `GET` primero para trabajar sobre el contenido fresco,
nunca asumir que `/tmp/page6.json` está actualizado.

---

### L-09 · Aurora Background — implementación CSS pura (no React)

**Contexto:** El usuario pidió integrar el componente React `AuroraBackground`
de Aceternity UI. El sitio es WordPress, no Next.js.

**Solución:** Traducir el efecto a CSS puro:
- `@keyframes aurora-move` con `background-position: 50%→350%` en 60 s.
- Capas de `repeating-linear-gradient` con colores navy/blue/cyan/indigo.
- `filter: blur(28px)` + `opacity: 0.18`.
- `mask-image: radial-gradient(ellipse 90% 80% at 50% 0%, ...)` para fade.
- `::after` con `mix-blend-mode: screen` para el brillo secundario.
- Respeta `prefers-reduced-motion` vía JS inline (`requestAnimationFrame`).

Clases usadas: `.hero-aurora` (wrapper), `.hero-aurora__layer` (capa animada).

---

### L-10 · No usar `<script type="module">` en páginas WordPress

**Contexto:** WordPress no maneja bien módulos ES en HTML personalizado.

**Regla:** Usar `<script>` tradicional con IIFE `(function(){ … })()`.

---

### L-11 · Zip del plugin — incluir solo el directorio interno

**Correcto:**
```bash
cd myrock-mail-engine
zip -r ../myrock-mail-engine.zip .
```

**Incorrecto:**
```bash
zip -r myrock-mail-engine.zip myrock-mail-engine/
# Crea un zip con un directorio de más → WP no activa correctamente
```

---

### L-12 · Formulario de suscripción — shortcode

Para incrustar el formulario en cualquier página/post de WordPress:
```
[mrme_form id="1"]
```

El ID corresponde al registro en la tabla `wp_mrme_forms`.
El handler de envío vive en `FormHandler.php` y `Shortcodes.php`.

---

### L-13 · Tags — color default gold

Los tags nuevos sin color explícito toman `#BFA26A` (gold de MyRock).
No cambiar este default sin actualizar la UI de admin también.

---

### L-14 · Application Password de WordPress

El Application Password `EjsR RFI5 2ipu l2r8 GaFh UlkS` pertenece al usuario
`admin` (ID 1) en myrock.com.mx. Es válido para WP REST API con Basic Auth.

**No confundir** con la contraseña de login del panel de WordPress.
Los Application Passwords se crean desde: WP Admin → Perfil → Application Passwords.

---

## Sitio (homepage CSS / diseño)

### L-15 · Variables CSS — no duplicar en línea

El sitio define todas las variables en `:root` dentro del bloque `<style>` de
la página ID 6. **No hardcodear colores en atributos `style=""`** salvo para
sobreescrituras puntuales. Usar siempre `var(--c-*)`.

### L-16 · Secciones eliminadas — logos horizontales

En marzo 2026 se eliminó la sección de "Variaciones del logo" del hero
(`.logo-card--wide`, `.logos-grid--wide`, `.logos-subtitle`).
Si algún agente ve referencias a esas clases en CSS, puede borrarlas con seguridad.

### L-17 · `f-string` con backslash en Python 3.11 es error de sintaxis

```python
# INCORRECTO en Python < 3.12:
f"found at: {content.find('<section id=\"hero\"')}"

# CORRECTO:
hero_id = 'section id="hero"'
f"found at: {content.find(hero_id)}"
```
