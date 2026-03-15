<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


/**
 * Admin Settings page for MyRock Mail Engine.
 */
class SettingsPage {

    /**
     * Option key used to store all plugin settings.
     *
     * @var string
     */
    private const OPTION_KEY = 'mrme_settings';

    /**
     * Render the settings page.
     *
     * @return void
     */
    public function render(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to view this page.', 'myrock-mail-engine' ) );
        }

        $settings = $this->get_settings();
        $notice   = $this->get_notice();

        $template = MRME_DIR . 'templates/admin/settings.php';
        if ( file_exists( $template ) ) {
            include $template;
            return;
        }

        // Fallback inline output.
        $this->render_inline( $settings, $notice );
    }

    /**
     * Render the settings form inline (fallback when template is missing).
     *
     * @param array      $settings Current settings array.
     * @param array|null $notice   Optional notice to display.
     * @return void
     */
    private function render_inline( array $settings, ?array $notice ): void {
        $mail_provider = $settings['mail_provider'] ?? 'wp_mail';

        echo '<div class="wrap">';
        echo '<h1>' . esc_html__( 'MyRock Mail Engine — Settings', 'myrock-mail-engine' ) . '</h1>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<form method="post" action="' . esc_url( admin_url( 'admin-post.php' ) ) . '">';
        echo '<input type="hidden" name="action" value="mrme_save_settings">';
        wp_nonce_field( 'mrme_save_settings' );

        // === Sender Settings ===.
        echo '<h2>' . esc_html__( 'Sender Details', 'myrock-mail-engine' ) . '</h2>';
        echo '<table class="form-table"><tbody>';

        printf(
            '<tr><th><label for="from_name">%s</label></th><td><input type="text" id="from_name" name="from_name" value="%s" class="regular-text"></td></tr>',
            esc_html__( 'From Name', 'myrock-mail-engine' ),
            esc_attr( $settings['from_name'] ?? '' )
        );

        printf(
            '<tr><th><label for="from_email">%s</label></th><td><input type="email" id="from_email" name="from_email" value="%s" class="regular-text"></td></tr>',
            esc_html__( 'From Email', 'myrock-mail-engine' ),
            esc_attr( $settings['from_email'] ?? '' )
        );

        printf(
            '<tr><th><label for="reply_to">%s</label></th><td><input type="email" id="reply_to" name="reply_to" value="%s" class="regular-text"></td></tr>',
            esc_html__( 'Reply-To Email', 'myrock-mail-engine' ),
            esc_attr( $settings['reply_to'] ?? '' )
        );

        echo '</tbody></table>';

        // === Mail Provider ===.
        echo '<h2>' . esc_html__( 'Mail Provider', 'myrock-mail-engine' ) . '</h2>';
        echo '<table class="form-table"><tbody>';

        echo '<tr><th><label for="mail_provider">' . esc_html__( 'Send Emails Via', 'myrock-mail-engine' ) . '</label></th><td>';
        echo '<select id="mail_provider" name="mail_provider">';
        $providers = [
            'wp_mail' => __( 'WordPress (wp_mail)', 'myrock-mail-engine' ),
            'smtp'    => __( 'SMTP', 'myrock-mail-engine' ),
        ];
        foreach ( $providers as $val => $label ) {
            printf( '<option value="%s" %s>%s</option>', esc_attr( $val ), selected( $mail_provider, $val, false ), esc_html( $label ) );
        }
        echo '</select>';
        echo '</td></tr>';

        echo '</tbody></table>';

        // === SMTP Settings (shown/hidden via JS) ===.
        printf( '<div id="mrme-smtp-settings"%s>', 'smtp' === $mail_provider ? '' : ' style="display:none;"' );
        echo '<h2>' . esc_html__( 'SMTP Configuration', 'myrock-mail-engine' ) . '</h2>';
        echo '<table class="form-table"><tbody>';

        printf(
            '<tr><th><label for="smtp_host">%s</label></th><td><input type="text" id="smtp_host" name="smtp_host" value="%s" class="regular-text" placeholder="smtp.example.com"></td></tr>',
            esc_html__( 'SMTP Host', 'myrock-mail-engine' ),
            esc_attr( $settings['smtp_host'] ?? '' )
        );

        printf(
            '<tr><th><label for="smtp_port">%s</label></th><td><input type="number" id="smtp_port" name="smtp_port" value="%s" class="small-text" placeholder="587"></td></tr>',
            esc_html__( 'SMTP Port', 'myrock-mail-engine' ),
            esc_attr( $settings['smtp_port'] ?? '587' )
        );

        echo '<tr><th><label for="smtp_encryption">' . esc_html__( 'Encryption', 'myrock-mail-engine' ) . '</label></th><td>';
        echo '<select id="smtp_encryption" name="smtp_encryption">';
        $encryptions = [
            'tls'  => __( 'TLS', 'myrock-mail-engine' ),
            'ssl'  => __( 'SSL', 'myrock-mail-engine' ),
            'none' => __( 'None', 'myrock-mail-engine' ),
        ];
        $current_encryption = $settings['smtp_encryption'] ?? 'tls';
        foreach ( $encryptions as $val => $label ) {
            printf( '<option value="%s" %s>%s</option>', esc_attr( $val ), selected( $current_encryption, $val, false ), esc_html( $label ) );
        }
        echo '</select>';
        echo '</td></tr>';

        printf(
            '<tr><th><label for="smtp_username">%s</label></th><td><input type="text" id="smtp_username" name="smtp_username" value="%s" class="regular-text" autocomplete="username"></td></tr>',
            esc_html__( 'SMTP Username', 'myrock-mail-engine' ),
            esc_attr( $settings['smtp_username'] ?? '' )
        );

        // Password: show placeholder text if already saved, allow updating via a new field.
        $has_password = ! empty( $settings['smtp_password'] );
        printf(
            '<tr><th><label for="smtp_password">%s</label></th><td>
                <input type="password" id="smtp_password" name="smtp_password" value="" class="regular-text" autocomplete="new-password" placeholder="%s">
                %s
             </td></tr>',
            esc_html__( 'SMTP Password', 'myrock-mail-engine' ),
            $has_password
                ? esc_attr__( 'Leave blank to keep existing password', 'myrock-mail-engine' )
                : esc_attr__( 'Enter SMTP password', 'myrock-mail-engine' ),
            $has_password
                ? '<p class="description">' . esc_html__( 'A password is currently saved. Enter a new one to replace it.', 'myrock-mail-engine' ) . '</p>'
                : ''
        );

        echo '</tbody></table>';
        echo '</div>'; // #mrme-smtp-settings.

        // === Plugin Behaviour ===.
        echo '<h2>' . esc_html__( 'Plugin Behaviour', 'myrock-mail-engine' ) . '</h2>';
        echo '<table class="form-table"><tbody>';

        printf(
            '<tr><th>%s</th><td><label><input type="checkbox" name="delete_on_uninstall" value="1" %s> %s</label><p class="description">%s</p></td></tr>',
            esc_html__( 'Uninstall Data', 'myrock-mail-engine' ),
            checked( $settings['delete_on_uninstall'] ?? false, true, false ),
            esc_html__( 'Delete all plugin data on uninstall', 'myrock-mail-engine' ),
            esc_html__( 'Warning: This will permanently remove all contacts, campaigns, and logs when the plugin is uninstalled.', 'myrock-mail-engine' )
        );

        echo '</tbody></table>';

        // Inline JS: toggle SMTP section when provider changes.
        echo '<script>
        (function(){
            var sel = document.getElementById("mail_provider");
            var box = document.getElementById("mrme-smtp-settings");
            if (sel && box) {
                sel.addEventListener("change", function(){
                    box.style.display = this.value === "smtp" ? "" : "none";
                });
            }
        })();
        </script>';

        submit_button( __( 'Save Settings', 'myrock-mail-engine' ) );

        echo '</form>';
        echo '</div>';
    }

    /**
     * Handle saving the settings form.
     *
     * @return void
     */
    public function handle_save(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_save_settings' );

        $redirect = admin_url( 'admin.php?page=mrme-settings' );

        // Retrieve existing settings so we can preserve the SMTP password if not replaced.
        $existing = $this->get_settings();

        // Sanitize all fields.
        $from_name       = isset( $_POST['from_name'] ) ? sanitize_text_field( wp_unslash( $_POST['from_name'] ) ) : '';
        $from_email      = isset( $_POST['from_email'] ) ? sanitize_email( wp_unslash( $_POST['from_email'] ) ) : '';
        $reply_to        = isset( $_POST['reply_to'] ) ? sanitize_email( wp_unslash( $_POST['reply_to'] ) ) : '';
        $mail_provider   = isset( $_POST['mail_provider'] ) ? sanitize_key( $_POST['mail_provider'] ) : 'wp_mail';
        $smtp_host       = isset( $_POST['smtp_host'] ) ? sanitize_text_field( wp_unslash( $_POST['smtp_host'] ) ) : '';
        $smtp_port       = isset( $_POST['smtp_port'] ) ? (int) $_POST['smtp_port'] : 587;
        $smtp_encryption = isset( $_POST['smtp_encryption'] ) ? sanitize_key( $_POST['smtp_encryption'] ) : 'tls';
        $smtp_username   = isset( $_POST['smtp_username'] ) ? sanitize_text_field( wp_unslash( $_POST['smtp_username'] ) ) : '';
        $smtp_password   = isset( $_POST['smtp_password'] ) ? wp_unslash( $_POST['smtp_password'] ) : ''; // phpcs:ignore WordPress.Security.ValidatedSanitizedInput

        $delete_on_uninstall = isset( $_POST['delete_on_uninstall'] ) ? true : false;

        // Validate mail_provider.
        if ( ! in_array( $mail_provider, [ 'wp_mail', 'smtp' ], true ) ) {
            $mail_provider = 'wp_mail';
        }

        // Validate SMTP encryption.
        if ( ! in_array( $smtp_encryption, [ 'tls', 'ssl', 'none' ], true ) ) {
            $smtp_encryption = 'tls';
        }

        // Clamp port to valid range.
        if ( $smtp_port < 1 || $smtp_port > 65535 ) {
            $smtp_port = 587;
        }

        // Only update the stored password if a new one was submitted; otherwise keep existing.
        if ( '' !== $smtp_password ) {
            // Sanitize but allow special characters typical in passwords.
            $smtp_password = sanitize_text_field( $smtp_password );
        } else {
            $smtp_password = $existing['smtp_password'] ?? '';
        }

        $settings = [
            'from_name'           => $from_name,
            'from_email'          => $from_email,
            'reply_to'            => $reply_to,
            'mail_provider'       => $mail_provider,
            'smtp_host'           => $smtp_host,
            'smtp_port'           => $smtp_port,
            'smtp_encryption'     => $smtp_encryption,
            'smtp_username'       => $smtp_username,
            'smtp_password'       => $smtp_password,
            'delete_on_uninstall' => $delete_on_uninstall,
        ];

        update_option( self::OPTION_KEY, $settings, false );

        wp_safe_redirect( add_query_arg( [ 'notice' => 'saved', 'notice_type' => 'success' ], $redirect ) );
        die();
    }

    /**
     * Retrieve the stored settings with defaults.
     *
     * @return array Settings array.
     */
    public function get_settings(): array {
        $defaults = [
            'from_name'           => get_bloginfo( 'name' ),
            'from_email'          => get_option( 'admin_email' ),
            'reply_to'            => '',
            'mail_provider'       => 'wp_mail',
            'smtp_host'           => '',
            'smtp_port'           => 587,
            'smtp_encryption'     => 'tls',
            'smtp_username'       => '',
            'smtp_password'       => '',
            'delete_on_uninstall' => false,
        ];

        $saved = get_option( self::OPTION_KEY, [] );

        if ( ! is_array( $saved ) ) {
            $saved = [];
        }

        return array_merge( $defaults, $saved );
    }

    /**
     * Retrieve and format a notice from query params.
     *
     * @return array|null Array with 'type' and 'message' keys, or null if none.
     */
    private function get_notice(): ?array {
        $notice_key  = isset( $_GET['notice'] ) ? sanitize_key( $_GET['notice'] ) : '';
        $notice_type = isset( $_GET['notice_type'] ) ? sanitize_key( $_GET['notice_type'] ) : 'success';

        if ( ! $notice_key ) {
            return null;
        }

        $messages = [
            'saved'       => __( 'Settings saved successfully.', 'myrock-mail-engine' ),
            'save_failed' => __( 'Failed to save settings. Please try again.', 'myrock-mail-engine' ),
        ];

        $message = isset( $messages[ $notice_key ] ) ? $messages[ $notice_key ] : '';

        if ( ! $message ) {
            return null;
        }

        return [
            'type'    => $notice_type,
            'message' => $message,
        ];
    }
}
