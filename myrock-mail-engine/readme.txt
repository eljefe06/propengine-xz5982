=== MyRock Mail Engine ===
Contributors: yanagui
Tags: email marketing, newsletter, contacts, campaigns, automation
Requires at least: 6.2
Tested up to: 6.9
Requires PHP: 8.0
Stable tag: 1.0.5
License: GPLv2 or later
License URI: https://www.gnu.org/licenses/gpl-2.0.html

Standalone email marketing plugin for WordPress. Manage contacts, send campaigns, and automate email flows — no external service required.

== Description ==

**MyRock Mail Engine** is a fully self-hosted email marketing solution for WordPress. No Mailchimp. No ActiveCampaign. No monthly fees for a third-party platform. Everything runs on your own server.

= Core Features =

* **Contact management** — Import via CSV, segment with lists and tags, full activity history per contact.
* **Email campaigns** — Full HTML editor, merge tags (`{{first_name}}`, `{{email}}`, `{{unsubscribe_url}}`), scheduling, and batch sending to avoid SMTP throttling.
* **Automations** — Event-triggered workflows: welcome sequences, follow-ups, conditional tagging, and configurable delays.
* **Subscription forms** — Embed forms anywhere with the `[mrme_form id="X"]` shortcode. Supports double opt-in.
* **Multiple mail providers** — Switch between `wp_mail` (default), a custom SMTP server, or Mailgun (Pro) without touching code.
* **REST API** — Full authenticated API to connect external systems. Endpoints for contacts, lists, campaigns, and forms.
* **WPForms integration** — Automatically capture leads from any WPForms form (Lite or Pro) directly into MRME contacts.
* **Unsubscribe handling** — Signed one-click unsubscribe URLs included in every campaign automatically.

= Free vs Pro =

The free version supports up to 500 contacts and 1 active campaign. The Pro license (available at [myrock.com.mx/plugin](https://myrock.com.mx/plugin/)) unlocks:

* Unlimited contacts
* Unlimited simultaneous campaigns
* Full automation workflows
* Mailgun provider integration
* Priority email support
* Automatic updates

= Privacy =

MyRock Mail Engine stores all contact data in your own WordPress database. No data is sent to third-party servers by default.

If you activate a **Pro license**, the plugin connects to `myrock.com.mx` once every 24 hours to validate the license key. The only data transmitted is your site URL and the license key. See the *External Services* section below.

== Installation ==

1. Upload the `myrock-mail-engine` folder to `/wp-content/plugins/`, or install directly from the WordPress plugin directory.
2. Activate the plugin from **Plugins → Installed Plugins**.
3. Go to **MyRock Mail → Settings** and configure your sender name, sender email, and mail provider (SMTP or Mailgun).
4. Create your first list under **MyRock Mail → Lists**, then add contacts or embed a subscription form.

== Frequently Asked Questions ==

= Do I need an external email service? =

No. By default the plugin uses WordPress's built-in `wp_mail()` function. For higher deliverability you can configure your own SMTP server or Mailgun (Pro).

= What happens when I reach 500 contacts on the free plan? =

New contacts will not be added until you either remove existing ones or upgrade to Pro.

= Does the plugin store data in the database? =

Yes. The plugin creates 11 custom tables (prefixed `{$wpdb->prefix}mrme_*`) on activation. All contacts, campaigns, logs, and settings are stored in your own WordPress database.

= Is the Pro license required to use the plugin? =

No. The free version is fully functional for small lists. Pro removes limits and adds advanced features.

= Can I use the plugin on multiple sites? =

The Pro license covers one WordPress installation. Contact hola@myrock.com.mx for multi-site pricing.

= How do I unsubscribe contacts? =

Every campaign includes a signed `{{unsubscribe_url}}` merge tag. Clicking it sets the contact status to `unsubscribed` immediately — no confirmation step required.

= Does it work with WPForms Lite? =

Yes. Enable the WPForms integration under **MyRock Mail → Settings → Integrations** and leads from any WPForms form will be saved automatically as contacts.

== External Services ==

This plugin connects to the following external services:

**1. Your configured mail provider (SMTP / Mailgun)**
When sending campaigns, the plugin delivers email through whichever provider you configure in Settings. No connection is made if you use the default `wp_mail()`.

* Mailgun API: https://www.mailgun.com — [Privacy Policy](https://www.mailgun.com/privacy-policy/) | [Terms of Service](https://www.mailgun.com/terms/)

**2. MyRock License Server (Pro license only)**
If you activate a Pro license key, the plugin sends your site URL and license key to `https://myrock.com.mx/wp-json/mrme-license/v1/validate` once every 24 hours to verify the license is valid. No other data is transmitted.

* Service: myrock.com.mx — [Privacy Policy](https://myrock.com.mx/privacidad/)

This connection is made **only when a Pro license key is entered**. Free users are not affected.

== Screenshots ==

1. Dashboard overview with send statistics.
2. Contact list with search, filters, and bulk actions.
3. Campaign editor with HTML preview and merge tag reference.
4. Automation workflow builder.
5. Settings page — mail provider configuration.

== Changelog ==

= 1.0.5 =
* Fix: Replace forbidden `move_uploaded_file()` with `wp_handle_upload()` in CSV import handler.
* Fix: Use `%i` identifier placeholder for table names in all direct DB queries (WP 6.2+).
* Fix: `Tested up to` updated to 6.9; `Requires at least` bumped to 6.2 to match `%i` usage.

= 1.0.4 =
* Fix: Plugin URI and Author URI were the same value — separated to comply with WordPress.org guidelines.

= 1.0.3 =
* Fix: Subscription form handler was registered on the wrong class (Shortcodes instead of FormHandler), causing a fatal error on form submit.
* Fix: Admin dropdown for manual license creation now reads prices from WP options instead of hardcoded values.
* New: WPForms integration — leads from WPForms forms are automatically saved as MRME contacts (Lite and Pro compatible).

= 1.0.2 =
* New: `POST /mrls/v1/checkout` REST endpoint creates MercadoPago payment preferences on the fly.
* New: Webhook handler now processes direct preference payments via `external_reference`.

= 1.0.1 =
* Internal: Auto-bump version hook added to development workflow.

= 1.0.0 =
* Initial release.
* Contact management with CSV import, lists, and tags.
* Campaign scheduling and batch sending.
* Automation engine with queue-based processing.
* Subscription forms with shortcode and double opt-in.
* REST API for contacts, lists, campaigns, and forms.
* wp_mail, SMTP, and Mailgun mail providers.

== Upgrade Notice ==

= 1.0.3 =
Fixes a fatal error on subscription form submit. Update recommended for all users.
