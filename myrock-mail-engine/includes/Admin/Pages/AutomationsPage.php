<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


use MyRock\MailEngine\Models\Automation;

/**
 * Admin Automations page for MyRock Mail Engine.
 */
class AutomationsPage {

    /**
     * Render the automations page.
     *
     * Dispatches to the edit form or list view based on query parameters.
     *
     * @return void
     */
    public function render(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to view this page.', 'myrock-mail-engine' ) );
        }

        $action = isset( $_GET['action'] ) ? sanitize_key( $_GET['action'] ) : 'list';
        $id     = isset( $_GET['id'] ) ? (int) $_GET['id'] : 0;
        $notice = $this->get_notice();

        if ( 'edit' === $action && $id > 0 ) {
            $automation = Automation::find( $id );

            if ( ! $automation ) {
                wp_die( esc_html__( 'Automation not found.', 'myrock-mail-engine' ) );
            }

            $steps = $this->get_steps( $id );
            $lists = $this->get_all_lists();
            $tags  = $this->get_all_tags();

            $template = MRME_DIR . 'templates/admin/automation-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            } else {
                $this->render_edit_form( $automation, $steps, $lists, $tags, $notice );
            }
            return;
        }

        if ( 'new' === $action ) {
            $automation = null;
            $steps      = [];
            $lists      = $this->get_all_lists();
            $tags       = $this->get_all_tags();

            $template = MRME_DIR . 'templates/admin/automation-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            } else {
                $this->render_edit_form( null, [], $lists, $tags, $notice );
            }
            return;
        }

        // Default: list view.
        $this->render_list( $notice );
    }

    /**
     * Render the automations list view.
     *
     * @param array|null $notice Optional notice to display.
     * @return void
     */
    private function render_list( ?array $notice ): void {
        global $wpdb;

        $table = $wpdb->prefix . 'mrme_automations';

        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $automations = $wpdb->get_results(
            "SELECT * FROM {$table} ORDER BY created_at DESC",
            ARRAY_A
        );

        if ( null === $automations ) {
            $automations = [];
        }

        $template = MRME_DIR . 'templates/admin/automations-list.php';
        if ( file_exists( $template ) ) {
            include $template;
            return;
        }

        // Fallback inline output.
        echo '<div class="wrap">';
        echo '<h1 class="wp-heading-inline">' . esc_html__( 'Automations', 'myrock-mail-engine' ) . '</h1>';
        echo '<a href="' . esc_url( admin_url( 'admin.php?page=mrme-automations&action=new' ) ) . '" class="page-title-action">' . esc_html__( 'Add New', 'myrock-mail-engine' ) . '</a>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<table class="wp-list-table widefat fixed striped"><thead><tr>';
        echo '<th>' . esc_html__( 'Name', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Trigger', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Status', 'myrock-mail-engine' ) . '</th>';
        echo '<th>' . esc_html__( 'Actions', 'myrock-mail-engine' ) . '</th>';
        echo '</tr></thead><tbody>';

        foreach ( $automations as $row ) {
            $edit_url   = admin_url( 'admin.php?page=mrme-automations&action=edit&id=' . (int) $row['id'] );
            $delete_url = wp_nonce_url(
                admin_url( 'admin-post.php?action=mrme_delete_automation&id=' . (int) $row['id'] ),
                'mrme_delete_automation_' . (int) $row['id']
            );

            echo '<tr>';
            printf( '<td><a href="%s">%s</a></td>', esc_url( $edit_url ), esc_html( $row['name'] ) );
            echo '<td>' . esc_html( $row['trigger'] ) . '</td>';
            echo '<td>' . esc_html( $row['status'] ) . '</td>';
            printf(
                '<td><a href="%s">%s</a> | <a href="%s" onclick="return confirm(\'%s\')">%s</a></td>',
                esc_url( $edit_url ),
                esc_html__( 'Edit', 'myrock-mail-engine' ),
                esc_url( $delete_url ),
                esc_js( __( 'Are you sure you want to delete this automation?', 'myrock-mail-engine' ) ),
                esc_html__( 'Delete', 'myrock-mail-engine' )
            );
            echo '</tr>';
        }

        echo '</tbody></table>';
        echo '</div>';
    }

    /**
     * Render the inline edit/create form (fallback when template is missing).
     *
     * @param Automation|object|array|null $automation Existing automation or null.
     * @param array                        $steps      Automation steps.
     * @param array                        $lists      All available lists.
     * @param array                        $tags       All available tags.
     * @param array|null                   $notice     Optional notice.
     * @return void
     */
    private function render_edit_form( $automation, array $steps, array $lists, array $tags, ?array $notice ): void {
        $is_new     = null === $automation;
        $page_title = $is_new ? __( 'Create Automation', 'myrock-mail-engine' ) : __( 'Edit Automation', 'myrock-mail-engine' );

        $automation_id  = $is_new ? 0 : ( is_array( $automation ) ? (int) $automation['id'] : (int) $automation->id );
        $name           = $is_new ? '' : ( is_array( $automation ) ? $automation['name'] : $automation->name );
        $trigger        = $is_new ? 'subscribe' : ( is_array( $automation ) ? $automation['trigger'] : $automation->trigger );
        $trigger_config = $is_new ? '{}' : ( is_array( $automation ) ? $automation['trigger_config'] : $automation->trigger_config );
        $status         = $is_new ? 'active' : ( is_array( $automation ) ? $automation['status'] : $automation->status );

        $available_triggers = [
            'subscribe'      => __( 'Contact Subscribes to List', 'myrock-mail-engine' ),
            'unsubscribe'    => __( 'Contact Unsubscribes', 'myrock-mail-engine' ),
            'tag_added'      => __( 'Tag Added to Contact', 'myrock-mail-engine' ),
            'tag_removed'    => __( 'Tag Removed from Contact', 'myrock-mail-engine' ),
            'form_submitted' => __( 'Form Submitted', 'myrock-mail-engine' ),
            'date_field'     => __( 'Contact Date Field', 'myrock-mail-engine' ),
        ];

        $available_step_types = [
            'email'  => __( 'Send Email', 'myrock-mail-engine' ),
            'wait'   => __( 'Wait / Delay', 'myrock-mail-engine' ),
            'tag'    => __( 'Add / Remove Tag', 'myrock-mail-engine' ),
            'list'   => __( 'Add / Remove from List', 'myrock-mail-engine' ),
            'webhook' => __( 'Webhook', 'myrock-mail-engine' ),
        ];

        echo '<div class="wrap">';
        echo '<h1>' . esc_html( $page_title ) . '</h1>';

        if ( $notice ) {
            printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
        }

        echo '<form method="post" action="' . esc_url( admin_url( 'admin-post.php' ) ) . '">';
        echo '<input type="hidden" name="action" value="mrme_save_automation">';
        printf( '<input type="hidden" name="automation_id" value="%d">', $automation_id );
        wp_nonce_field( 'mrme_save_automation' );

        echo '<table class="form-table"><tbody>';

        // Name.
        printf(
            '<tr><th><label for="automation_name">%s</label></th><td><input type="text" id="automation_name" name="name" value="%s" class="regular-text" required></td></tr>',
            esc_html__( 'Automation Name', 'myrock-mail-engine' ),
            esc_attr( $name )
        );

        // Trigger.
        echo '<tr><th><label for="automation_trigger">' . esc_html__( 'Trigger', 'myrock-mail-engine' ) . '</label></th><td>';
        echo '<select id="automation_trigger" name="trigger">';
        foreach ( $available_triggers as $val => $label ) {
            printf( '<option value="%s" %s>%s</option>', esc_attr( $val ), selected( $trigger, $val, false ), esc_html( $label ) );
        }
        echo '</select>';
        echo '</td></tr>';

        // Trigger Config (JSON).
        printf(
            '<tr><th><label for="trigger_config">%s</label></th><td><textarea id="trigger_config" name="trigger_config" rows="4" class="large-text code">%s</textarea><p class="description">%s</p></td></tr>',
            esc_html__( 'Trigger Configuration (JSON)', 'myrock-mail-engine' ),
            esc_textarea( $trigger_config ),
            esc_html__( 'e.g. {"list_id": 1} for subscribe trigger.', 'myrock-mail-engine' )
        );

        // Status.
        echo '<tr><th><label for="automation_status">' . esc_html__( 'Status', 'myrock-mail-engine' ) . '</label></th><td>';
        echo '<select id="automation_status" name="status">';
        $statuses = [ 'active' => __( 'Active', 'myrock-mail-engine' ), 'paused' => __( 'Paused', 'myrock-mail-engine' ) ];
        foreach ( $statuses as $val => $label ) {
            printf( '<option value="%s" %s>%s</option>', esc_attr( $val ), selected( $status, $val, false ), esc_html( $label ) );
        }
        echo '</select>';
        echo '</td></tr>';

        echo '</tbody></table>';

        // Steps.
        echo '<h2>' . esc_html__( 'Automation Steps', 'myrock-mail-engine' ) . '</h2>';
        echo '<p class="description">' . esc_html__( 'Define steps as JSON. Each step must have a "type" and "config" key.', 'myrock-mail-engine' ) . '</p>';

        echo '<div id="mrme-automation-steps">';

        if ( ! empty( $steps ) ) {
            foreach ( $steps as $index => $step ) {
                $step_type   = is_array( $step ) ? $step['type'] : $step->type;
                $step_config = is_array( $step ) ? $step['config'] : $step->config;
                $step_order  = is_array( $step ) ? (int) $step['step_order'] : (int) $step->step_order;
                $step_id     = is_array( $step ) ? (int) $step['id'] : (int) $step->id;

                echo '<div class="mrme-step" data-index="' . esc_attr( $index ) . '">';
                printf( '<input type="hidden" name="steps[%d][id]" value="%d">', $index, $step_id );
                printf( '<input type="hidden" name="steps[%d][step_order]" value="%d">', $index, $step_order );

                echo '<select name="steps[' . esc_attr( $index ) . '][type]">';
                foreach ( $available_step_types as $val => $label ) {
                    printf( '<option value="%s" %s>%s</option>', esc_attr( $val ), selected( $step_type, $val, false ), esc_html( $label ) );
                }
                echo '</select>';

                printf(
                    '<textarea name="steps[%d][config]" rows="3" class="large-text code">%s</textarea>',
                    $index,
                    esc_textarea( is_array( $step_config ) ? wp_json_encode( $step_config ) : $step_config )
                );
                echo '</div>';
            }
        } else {
            // Default first step.
            echo '<div class="mrme-step" data-index="0">';
            echo '<input type="hidden" name="steps[0][id]" value="0">';
            echo '<input type="hidden" name="steps[0][step_order]" value="1">';

            echo '<select name="steps[0][type]">';
            foreach ( $available_step_types as $val => $label ) {
                printf( '<option value="%s">%s</option>', esc_attr( $val ), esc_html( $label ) );
            }
            echo '</select>';

            echo '<textarea name="steps[0][config]" rows="3" class="large-text code">{}</textarea>';
            echo '</div>';
        }

        echo '</div>'; // #mrme-automation-steps.

        echo '<p>';
        echo '<button type="button" class="button" id="mrme-add-step">' . esc_html__( '+ Add Step', 'myrock-mail-engine' ) . '</button>';
        echo '</p>';

        submit_button( $is_new ? __( 'Create Automation', 'myrock-mail-engine' ) : __( 'Update Automation', 'myrock-mail-engine' ) );

        echo '</form>';
        echo '</div>';
    }

    /**
     * Handle saving an automation and its steps.
     *
     * @return void
     */
    public function handle_save(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_save_automation' );

        global $wpdb;

        $automation_id  = isset( $_POST['automation_id'] ) ? (int) $_POST['automation_id'] : 0;
        $name           = isset( $_POST['name'] ) ? sanitize_text_field( wp_unslash( $_POST['name'] ) ) : '';
        $trigger        = isset( $_POST['trigger'] ) ? sanitize_key( $_POST['trigger'] ) : 'subscribe';
        $trigger_config = isset( $_POST['trigger_config'] ) ? wp_unslash( $_POST['trigger_config'] ) : '{}'; // phpcs:ignore WordPress.Security.ValidatedSanitizedInput
        $status         = isset( $_POST['status'] ) ? sanitize_key( $_POST['status'] ) : 'active';

        $redirect = admin_url( 'admin.php?page=mrme-automations' );

        if ( ! $name ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'name_required', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        // Validate trigger config JSON.
        $trigger_config_decoded = json_decode( $trigger_config, true );
        if ( null === $trigger_config_decoded ) {
            $trigger_config = '{}';
        } else {
            $trigger_config = wp_json_encode( $trigger_config_decoded );
        }

        if ( ! in_array( $status, [ 'active', 'paused' ], true ) ) {
            $status = 'active';
        }

        $data = [
            'name'           => $name,
            'trigger'        => $trigger,
            'trigger_config' => $trigger_config,
            'status'         => $status,
        ];

        if ( $automation_id > 0 ) {
            $result = Automation::update( $automation_id, $data );
            $action = 'updated';
        } else {
            $result        = Automation::create( $data );
            $action        = 'created';

            if ( $result ) {
                $automation_id = (int) $result;
            }
        }

        if ( false === $result || is_wp_error( $result ) ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'save_failed', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        // Save steps.
        if ( $automation_id > 0 ) {
            $this->save_steps( $automation_id );
        }

        $redirect = admin_url( 'admin.php?page=mrme-automations&action=edit&id=' . $automation_id );
        wp_safe_redirect( add_query_arg( [ 'notice' => $action, 'notice_type' => 'success' ], $redirect ) );
        die();
    }

    /**
     * Save automation steps for a given automation.
     *
     * Deletes existing steps and re-inserts all submitted steps.
     *
     * @param int $automation_id Automation ID.
     * @return void
     */
    private function save_steps( int $automation_id ): void {
        global $wpdb;

        $steps_table = $wpdb->prefix . 'mrme_automation_steps';

        // Delete all existing steps for this automation.
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $wpdb->delete( $steps_table, [ 'automation_id' => $automation_id ], [ '%d' ] );

        if ( ! isset( $_POST['steps'] ) || ! is_array( $_POST['steps'] ) ) {
            return;
        }

        $steps = $_POST['steps']; // phpcs:ignore WordPress.Security.ValidatedSanitizedInput

        foreach ( $steps as $index => $step ) {
            $step_type   = isset( $step['type'] ) ? sanitize_key( $step['type'] ) : 'email';
            $step_config = isset( $step['config'] ) ? wp_unslash( $step['config'] ) : '{}'; // phpcs:ignore WordPress.Security.ValidatedSanitizedInput
            $step_order  = isset( $step['step_order'] ) ? (int) $step['step_order'] : ( (int) $index + 1 );

            // Validate step config JSON.
            $config_decoded = json_decode( $step_config, true );
            if ( null === $config_decoded ) {
                $step_config = '{}';
            } else {
                $step_config = wp_json_encode( $config_decoded );
            }

            // phpcs:ignore WordPress.DB.DirectDatabaseQuery
            $wpdb->insert(
                $steps_table,
                [
                    'automation_id' => $automation_id,
                    'type'          => $step_type,
                    'config'        => $step_config,
                    'step_order'    => $step_order,
                    'created_at'    => current_time( 'mysql' ),
                ],
                [ '%d', '%s', '%s', '%d', '%s' ]
            );
        }
    }

    /**
     * Fetch steps for an automation.
     *
     * @param int $automation_id Automation ID.
     * @return array
     */
    private function get_steps( int $automation_id ): array {
        global $wpdb;
        $table = $wpdb->prefix . 'mrme_automation_steps';

        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $results = $wpdb->get_results(
            $wpdb->prepare(
                "SELECT * FROM {$table} WHERE automation_id = %d ORDER BY step_order ASC",
                $automation_id
            ),
            ARRAY_A
        );

        return $results ?: [];
    }

    /**
     * Fetch all mailing lists.
     *
     * @return array
     */
    private function get_all_lists(): array {
        global $wpdb;
        $table = $wpdb->prefix . 'mrme_lists';
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $results = $wpdb->get_results( "SELECT id, name FROM {$table} ORDER BY name ASC", ARRAY_A );
        return $results ?: [];
    }

    /**
     * Fetch all tags.
     *
     * @return array
     */
    private function get_all_tags(): array {
        global $wpdb;
        $table = $wpdb->prefix . 'mrme_tags';
        // phpcs:ignore WordPress.DB.DirectDatabaseQuery
        $results = $wpdb->get_results( "SELECT id, name FROM {$table} ORDER BY name ASC", ARRAY_A );
        return $results ?: [];
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
            'created'       => __( 'Automation created successfully.', 'myrock-mail-engine' ),
            'updated'       => __( 'Automation updated successfully.', 'myrock-mail-engine' ),
            'deleted'       => __( 'Automation deleted successfully.', 'myrock-mail-engine' ),
            'save_failed'   => __( 'Failed to save automation. Please try again.', 'myrock-mail-engine' ),
            'delete_failed' => __( 'Failed to delete automation. Please try again.', 'myrock-mail-engine' ),
            'name_required' => __( 'Automation name is required.', 'myrock-mail-engine' ),
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
