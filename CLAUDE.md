# CLAUDE.md — MyRock · propengine-xz5982

Context file for Claude Code agents working on this repository.
Read this before touching any file.

---

## Repo at a glance

| Item | Value |
|---|---|
| Site | https://myrock.com.mx |
| Stack | WordPress 6.x (Docker) · PHP 8.0+ · vanilla HTML/CSS/JS |
| Branch principal | `claude/install-google-stitch-mcp-zt9ob` |
| WP Admin credentials | user `admin` · Application Password en variable de entorno `WP_APP_PASSWORD` (o pedir al operador) |
| WP REST base | `https://myrock.com.mx/wp-json/wp/v2/` |
| MRME REST base | `https://myrock.com.mx/wp-json/mrme/v1/` |

> **No es un proyecto React/Next.js.** El frontend del sitio es HTML/CSS/JS puro
> embebido en páginas de WordPress via el editor de bloques (HTML personalizado).
> No uses Tailwind, shadcn ni ningún framework JS salvo para tareas internas de tooling.

---

## Estructura del repositorio

```
propengine-xz5982/
├── myrock/                    # CSS compartido estático del sitio
│   ├── style.css              # Estilos globales (variables, layout, secciones)
│   └── producto.css           # Estilos página de producto
├── myrock-theme/              # Child theme de WordPress (Twenty Twenty-Five)
│   ├── style.css
│   ├── functions.php
│   ├── index.php / single.php / archive.php / search.php
│   └── template-parts/        # Header, footer, newsletter CTA, post CTA
├── myrock-mail-engine/        # Plugin principal — ver sección detallada abajo
├── myrock-mail-engine.zip     # Zip listo para subir a WP (regenerar con: zip -r)
├── myrock-theme.zip           # Zip del child theme
├── deploy-myrock.sh           # Script de deploy al Docker de producción
├── manual-myrock-mail-engine.html  # Manual HTML completo del plugin
└── codex.py                   # Utilidades Python de apoyo
```

---

## Sitio web (myrock.com.mx)

El contenido de la homepage vive en la **página de WordPress con ID 6**.
Se edita vía WP REST API (no hay archivos locales que lo representen):

```bash
# Leer contenido actual
curl -s -u "admin:APP_PASSWORD" \
  "https://myrock.com.mx/wp-json/wp/v2/pages/6?context=edit"

# Actualizar contenido
curl -s -u "admin:APP_PASSWORD" \
  -X POST "https://myrock.com.mx/wp-json/wp/v2/pages/6" \
  -H "Content-Type: application/json" \
  -d '{"content": "NUEVO_HTML"}'
```

### Paleta de colores MyRock

```css
--c-bg:      #FFFFFF
--c-text-1:  #0F172A   /* near black */
--c-gold:    #1B2980   /* navy — color primario del logo */
--c-navy:    #1B2980
--c-blue:    #2563EB
--c-cyan:    #06B6D4
--c-border:  #DBEAFE
```

### Tipografía

- Serif: `Instrument Serif` (títulos, hero)
- Sans: `Inter`
- Mono: `DM Mono`

### Secciones activas en la homepage (ID 6)

`#hero` → `#cases` → `#services` → `#process` → `#manifesto` → `#pricing` → `#newsletter`

### Aurora Background en el hero

El hero (`#hero`) tiene una capa de efecto aurora animado implementada como
CSS puro (sin React). La clase raíz es `.hero-aurora` con un hijo `.hero-aurora__layer`.
La animación se llama `aurora-move` (60 s, linear, infinite).

---

## Plugin: MyRock Mail Engine (MRME)

Plugin WordPress de email marketing standalone. **No depende de ningún servicio
externo** salvo el proveedor SMTP configurado por el administrador.

### Instalación / activación

1. Subir `myrock-mail-engine.zip` desde WP Admin → Plugins → Añadir nuevo.
2. Activar el plugin → crea automáticamente las 11 tablas de BD + datos semilla.
3. Ir a **MyRock Mail → Configuración** y definir From Name, From Email y SMTP.

### Namespace PHP

```
MyRock\MailEngine\
```

Autoloader PSR-4 simple desde `myrock-mail-engine.php`.
Prefijo de tablas BD: `{$wpdb->prefix}mrme_*` (ej. `wp_mrme_contacts`).
Prefijo de opciones WP: `mrme_*`.
Prefijo de hooks WP: `mrme_*`.

### Árbol de archivos clave

```
myrock-mail-engine/
├── myrock-mail-engine.php          # Entry point, constantes, autoloader
├── uninstall.php
├── includes/
│   ├── Core/
│   │   ├── Plugin.php              # Singleton bootstrap — registra todos los hooks
│   │   ├── Loader.php              # Agrega/ejecuta hooks de WP
│   │   ├── Activator.php           # create_tables() + seed_defaults() al activar
│   │   └── Deactivator.php         # Limpia cron al desactivar
│   ├── Database/
│   │   └── Schema.php              # dbDelta() para las 11 tablas + seed inicial
│   ├── Models/                     # Active-Record ligero sobre $wpdb
│   │   ├── Contact.php
│   │   ├── Campaign.php
│   │   ├── Form.php
│   │   ├── Tag.php
│   │   ├── Automation.php
│   │   └── SendLog.php
│   ├── Services/
│   │   ├── ContactService.php      # create/update, unsubscribe URL, CSV import
│   │   ├── CampaignService.php     # schedule/send_now/send_batch/replace_placeholders
│   │   ├── AutomationService.php   # trigger_for_contact / process_pending_steps
│   │   └── CsvImporter.php
│   ├── Mail/
│   │   ├── MailManager.php         # Fachada estática: resuelve proveedor y envía
│   │   ├── MailMessage.php         # DTO fluido (setTo/setSubject/setHtml/…)
│   │   └── Providers/
│   │       ├── ProviderInterface.php
│   │       ├── WpMailProvider.php  # wp_mail() fallback
│   │       └── SmtpProvider.php    # PHPMailer directo
│   ├── Admin/
│   │   ├── Admin.php               # Registra menús, assets y handlers POST
│   │   └── Pages/                  # Una clase por pantalla del admin
│   ├── Api/
│   │   └── RestApi.php             # Rutas REST bajo mrme/v1
│   ├── Public/
│   │   ├── Shortcodes.php          # [mrme_form id="X"]
│   │   └── FormHandler.php         # Unsubscribe + double opt-in confirm
│   └── Services/
└── templates/
    └── admin/                      # Vistas PHP de las pantallas del plugin
```

### Base de datos — 11 tablas

| Tabla | Propósito |
|---|---|
| `mrme_contacts` | Suscriptores. Campos: email, first/last name, phone, company, status, source, ip, notes, meta |
| `mrme_lists` | Listas de correo |
| `mrme_contact_lists` | Pivot contacto ↔ lista |
| `mrme_tags` | Etiquetas de segmentación |
| `mrme_contact_tags` | Pivot contacto ↔ tag |
| `mrme_campaigns` | Campañas: subject, html, text, from, status, estadísticas |
| `mrme_send_logs` | Un row por contacto por campaña (pending→sent/failed) |
| `mrme_events` | Aperturas, clics, bajas, rebotes |
| `mrme_forms` | Formularios de suscripción con campos JSON |
| `mrme_automations` | Definición de automatizaciones (trigger + status) |
| `mrme_automation_steps` | Pasos de cada automatización (email/wait/tag/untag/condition) |
| `mrme_automation_queue` | Cola de ejecución pendiente de pasos |

### Estados de contacto

`subscribed` | `unsubscribed` | `pending` | `bounced` | `complained`

### Merge tags disponibles en plantillas

`{{first_name}}` `{{last_name}}` `{{email}}` `{{company}}` `{{unsubscribe_url}}`

### WP Cron

Dos eventos registrados cada 5 minutos:
- `mrme_process_campaigns` → `CampaignService::process_scheduled()`
- `mrme_process_automations` → `AutomationService::process_pending_steps()`

### REST API (mrme/v1)

Autenticación: usuario admin con `manage_options` O header `X-MRME-Key`.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/contacts` | Lista paginada (search, status, list_id, per_page, page) |
| POST | `/contacts` | Crear contacto |
| GET | `/contacts/{id}` | Detalle |
| PUT | `/contacts/{id}` | Actualizar |
| DELETE | `/contacts/{id}` | Eliminar |
| GET | `/lists` | Todas las listas |
| GET | `/campaigns` | Campañas (status, per_page, page) |
| GET | `/forms` | Todos los formularios |

### Proveedores de correo

Configurables desde WP Admin → MyRock Mail → Configuración:
- `wp_mail` — usa la función nativa de WordPress (default)
- `smtp` — PHPMailer con host/port/auth/TLS configurables

---

## Cómo hacer deploy

```bash
# Requiere acceso SSH al servidor con Docker
bash deploy-myrock.sh
```

El script clona el branch actual, copia los CSS estáticos al volumen Docker
(`/var/lib/docker/volumes/myrock-stack_wordpress_data/_data`) y limpia HTMLs
estáticos huérfanos.

Para actualizar **solo el plugin**:
1. Regenerar zip: `cd myrock-mail-engine && zip -r ../myrock-mail-engine.zip .`
2. Subir desde WP Admin o vía SCP al servidor.

---

## Reglas para agentes

1. **No crear archivos React/Next.js** en este repo. El frontend es WP + HTML/CSS puro.
2. **No modificar tablas de BD directamente** — usar `Schema.php` con `dbDelta()`.
3. **El branch de trabajo es `claude/install-google-stitch-mcp-zt9ob`**. No pushear a `main`.
4. Cambios en la homepage → editar vía WP REST API (página ID 6), no hay archivo local.
5. Cambios en el plugin → editar los `.php` locales y regenerar el `.zip`.
6. Mantener el prefijo `mrme_` en todas las tablas, opciones, hooks y transients nuevos.
7. Respetar el autoloader PSR-4: namespace `MyRock\MailEngine\Carpeta\Clase` → archivo `includes/Carpeta/Clase.php`.
