<?php
namespace MyRock\MailEngine\Admin\Pages;

defined('ABSPATH') || exit;


use MyRock\MailEngine\Models\Contact;
use MyRock\MailEngine\Services\ContactService;

/**
 * Admin Contacts page for MyRock Mail Engine.
 */
class ContactsPage {

    /**
     * Number of contacts to display per page.
     *
     * @var int
     */
    private const PER_PAGE = 20;

    /**
     * Render the contacts page.
     *
     * Dispatches to the edit form, new form, or contacts list depending on
     * query parameters.
     *
     * @return void
     */
    public function render(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to view this page.', 'myrock-mail-engine' ) );
        }

        $action = isset( $_GET['action'] ) ? sanitize_key( $_GET['action'] ) : 'list';
        $id     = isset( $_GET['id'] ) ? (int) $_GET['id'] : 0;

        if ( 'edit' === $action && $id > 0 ) {
            $contact = Contact::find( $id );

            if ( ! $contact ) {
                wp_die( esc_html__( 'Contact not found.', 'myrock-mail-engine' ) );
            }

            $notice = $this->get_notice();

            $template = MRME_DIR . 'templates/admin/contact-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            }
            return;
        }

        if ( 'new' === $action ) {
            $contact = null;
            $notice  = $this->get_notice();

            $template = MRME_DIR . 'templates/admin/contact-edit.php';
            if ( file_exists( $template ) ) {
                include $template;
            }
            return;
        }

        // Default: list view.
        $this->render_list();
    }

    /**
     * Render the contacts list view.
     *
     * @return void
     */
    private function render_list(): void {
        global $wpdb;

        $table        = $wpdb->prefix . 'mrme_contacts';
        $search       = isset( $_GET['search'] ) ? sanitize_text_field( wp_unslash( $_GET['search'] ) ) : '';
        $status       = isset( $_GET['status'] ) ? sanitize_key( $_GET['status'] ) : '';
        $current_page = isset( $_GET['paged'] ) ? max( 1, (int) $_GET['paged'] ) : 1;
        $offset       = ( $current_page - 1 ) * self::PER_PAGE;

        $where_clauses = [];
        $placeholders  = [];

        if ( $search ) {
            $like            = '%' . $wpdb->esc_like( $search ) . '%';
            $where_clauses[] = '(email LIKE %s OR first_name LIKE %s OR last_name LIKE %s)';
            $placeholders[]  = $like;
            $placeholders[]  = $like;
            $placeholders[]  = $like;
        }

        if ( $status ) {
            $where_clauses[] = 'status = %s';
            $placeholders[]  = $status;
        }

        $where_sql = $where_clauses ? 'WHERE ' . implode( ' AND ', $where_clauses ) : '';

        // Use %i placeholder (WP 6.2+) for the table name to satisfy WPCS DirectDB checks.
        // phpcs:ignore WordPress.DB.PreparedSQL.NotPrepared
        $count_sql = 'SELECT COUNT(*) FROM %i ' . $where_sql;
        // phpcs:ignore WordPress.DB.PreparedSQL.NotPrepared
        $list_sql  = 'SELECT * FROM %i ' . $where_sql . ' ORDER BY id DESC LIMIT %d OFFSET %d';

        if ( $placeholders ) {
            // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.NotPrepared
            $total_items = (int) $wpdb->get_var( $wpdb->prepare( $count_sql, array_merge( [ $table ], $placeholders ) ) );

            // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.NotPrepared
            $contacts = $wpdb->get_results( $wpdb->prepare( $list_sql, array_merge( [ $table ], $placeholders, [ self::PER_PAGE, $offset ] ) ), ARRAY_A );
        } else {
            // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.NotPrepared
            $total_items = (int) $wpdb->get_var( $wpdb->prepare( $count_sql, $table ) );
            // phpcs:ignore WordPress.DB.DirectDatabaseQuery, WordPress.DB.PreparedSQL.NotPrepared
            $contacts = $wpdb->get_results( $wpdb->prepare( $list_sql, $table, self::PER_PAGE, $offset ), ARRAY_A );
        }

        if ( null === $contacts ) {
            $contacts = [];
        }

        $total_pages = (int) ceil( $total_items / self::PER_PAGE );
        $notice      = $this->get_notice();

        $template = MRME_DIR . 'templates/admin/contacts-list.php';
        if ( file_exists( $template ) ) {
            include $template;
        } else {
            // Fallback inline output when template file is not present.
            echo '<div class="wrap">';
            echo '<h1 class="wp-heading-inline">' . esc_html__( 'Contacts', 'myrock-mail-engine' ) . '</h1>';
            echo '<a href="' . esc_url( admin_url( 'admin.php?page=mrme-contacts&action=new' ) ) . '" class="page-title-action">' . esc_html__( 'Add New', 'myrock-mail-engine' ) . '</a>';

            if ( $notice ) {
                printf( '<div class="notice notice-%s is-dismissible"><p>%s</p></div>', esc_attr( $notice['type'] ), esc_html( $notice['message'] ) );
            }

            echo '<p>' . esc_html( sprintf( _n( '%d contact found.', '%d contacts found.', $total_items, 'myrock-mail-engine' ), $total_items ) ) . '</p>';
            echo '<table class="wp-list-table widefat fixed striped"><thead><tr>';
            echo '<th>' . esc_html__( 'Email', 'myrock-mail-engine' ) . '</th>';
            echo '<th>' . esc_html__( 'First Name', 'myrock-mail-engine' ) . '</th>';
            echo '<th>' . esc_html__( 'Last Name', 'myrock-mail-engine' ) . '</th>';
            echo '<th>' . esc_html__( 'Status', 'myrock-mail-engine' ) . '</th>';
            echo '<th>' . esc_html__( 'Actions', 'myrock-mail-engine' ) . '</th>';
            echo '</tr></thead><tbody>';

            foreach ( $contacts as $row ) {
                $edit_url   = admin_url( 'admin.php?page=mrme-contacts&action=edit&id=' . (int) $row['id'] );
                $delete_url = wp_nonce_url(
                    admin_url( 'admin-post.php?action=mrme_delete_contact&id=' . (int) $row['id'] ),
                    'mrme_delete_contact_' . (int) $row['id']
                );

                echo '<tr>';
                printf( '<td><a href="%s">%s</a></td>', esc_url( $edit_url ), esc_html( $row['email'] ) );
                echo '<td>' . esc_html( $row['first_name'] ) . '</td>';
                echo '<td>' . esc_html( $row['last_name'] ) . '</td>';
                echo '<td>' . esc_html( $row['status'] ) . '</td>';
                printf(
                    '<td><a href="%s">%s</a> | <a href="%s" onclick="return confirm(\'%s\')">%s</a></td>',
                    esc_url( $edit_url ),
                    esc_html__( 'Edit', 'myrock-mail-engine' ),
                    esc_url( $delete_url ),
                    esc_js( __( 'Are you sure you want to delete this contact?', 'myrock-mail-engine' ) ),
                    esc_html__( 'Delete', 'myrock-mail-engine' )
                );
                echo '</tr>';
            }

            echo '</tbody></table>';
            echo '</div>';
        }
    }

    /**
     * Handle saving a contact (create or update).
     *
     * @return void
     */
    public function handle_save(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        $contact_id  = isset( $_POST['contact_id'] ) ? (int) $_POST['contact_id'] : 0;

        check_admin_referer( 'mrme_save_contact_' . $contact_id );
        $email       = isset( $_POST['email'] ) ? sanitize_email( wp_unslash( $_POST['email'] ) ) : '';
        $first_name  = isset( $_POST['first_name'] ) ? sanitize_text_field( wp_unslash( $_POST['first_name'] ) ) : '';
        $last_name   = isset( $_POST['last_name'] ) ? sanitize_text_field( wp_unslash( $_POST['last_name'] ) ) : '';
        $status      = isset( $_POST['status'] ) ? sanitize_key( $_POST['status'] ) : 'subscribed';
        $phone       = isset( $_POST['phone'] ) ? sanitize_text_field( wp_unslash( $_POST['phone'] ) ) : '';
        $meta        = isset( $_POST['meta'] ) && is_array( $_POST['meta'] ) ? array_map( 'sanitize_text_field', wp_unslash( $_POST['meta'] ) ) : [];

        // Validate required fields.
        if ( ! $email || ! is_email( $email ) ) {
            $redirect = admin_url( 'admin.php?page=mrme-contacts' );

            if ( $contact_id > 0 ) {
                $redirect = admin_url( 'admin.php?page=mrme-contacts&action=edit&id=' . $contact_id );
            } else {
                $redirect = admin_url( 'admin.php?page=mrme-contacts&action=new' );
            }

            wp_safe_redirect( add_query_arg( [ 'notice' => 'invalid_email', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $data = [
            'email'      => $email,
            'first_name' => $first_name,
            'last_name'  => $last_name,
            'status'     => $status,
            'phone'      => $phone,
            'meta'       => wp_json_encode( $meta ),
        ];

        if ( $contact_id > 0 ) {
            $result = Contact::update( $contact_id, $data );
            $action = 'updated';
        } else {
            $result = Contact::create( $data );
            $action = 'created';

            if ( $result ) {
                $contact_id = $result;
            }
        }

        if ( false === $result || is_wp_error( $result ) ) {
            $redirect = $contact_id > 0
                ? admin_url( 'admin.php?page=mrme-contacts&action=edit&id=' . $contact_id )
                : admin_url( 'admin.php?page=mrme-contacts&action=new' );

            wp_safe_redirect( add_query_arg( [ 'notice' => 'save_failed', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $redirect = admin_url( 'admin.php?page=mrme-contacts&action=edit&id=' . (int) $contact_id );
        wp_safe_redirect( add_query_arg( [ 'notice' => $action, 'notice_type' => 'success' ], $redirect ) );
        die();
    }

    /**
     * Handle deleting a contact.
     *
     * @return void
     */
    public function handle_delete(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        $id = isset( $_GET['id'] ) ? (int) $_GET['id'] : 0;

        if ( ! $id ) {
            wp_die( esc_html__( 'Invalid contact ID.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_delete_contact_' . $id );

        $result = Contact::delete( $id );

        $redirect = admin_url( 'admin.php?page=mrme-contacts' );

        if ( false === $result ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'delete_failed', 'notice_type' => 'error' ], $redirect ) );
        } else {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'deleted', 'notice_type' => 'success' ], $redirect ) );
        }

        die();
    }

    /**
     * Handle importing contacts from a CSV file.
     *
     * @return void
     */
    public function handle_import(): void {
        if ( ! current_user_can( 'manage_options' ) ) {
            wp_die( esc_html__( 'You do not have permission to perform this action.', 'myrock-mail-engine' ) );
        }

        check_admin_referer( 'mrme_import_csv' );

        $redirect = admin_url( 'admin.php?page=mrme-contacts' );

        // Validate uploaded file.
        if (
            ! isset( $_FILES['csv_file'] )
            || UPLOAD_ERR_OK !== $_FILES['csv_file']['error']
        ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'import_no_file', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        // Use wp_handle_upload() — move_uploaded_file() is forbidden on WP.org.
        require_once ABSPATH . 'wp-admin/includes/file.php';

        $overrides = [
            'test_form' => false,
            'mimes'     => [ 'csv' => 'text/csv' ],
        ];

        // phpcs:ignore WordPress.Security.ValidatedSanitizedInput.InputNotSanitized
        $uploaded = wp_handle_upload( $_FILES['csv_file'], $overrides );

        if ( isset( $uploaded['error'] ) ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'import_upload_failed', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $file_name = isset( $uploaded['file'] ) ? $uploaded['file'] : '';
        $extension = strtolower( pathinfo( $file_name, PATHINFO_EXTENSION ) );

        if ( 'csv' !== $extension ) {
            wp_delete_file( $file_name );
            wp_safe_redirect( add_query_arg( [ 'notice' => 'import_invalid_type', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $tmp_file = $file_name;

        $list_id = isset( $_POST['list_id'] ) ? (int) $_POST['list_id'] : 0;

        $service = new ContactService();
        $result  = $service->import_csv( $tmp_file, $list_id );

        // Remove the temp file.
        if ( file_exists( $tmp_file ) ) {
            wp_delete_file( $tmp_file );
        }

        if ( is_wp_error( $result ) ) {
            wp_safe_redirect( add_query_arg( [ 'notice' => 'import_failed', 'notice_type' => 'error' ], $redirect ) );
            die();
        }

        $imported = isset( $result['imported'] ) ? (int) $result['imported'] : 0;
        $skipped  = isset( $result['skipped'] ) ? (int) $result['skipped'] : 0;

        wp_safe_redirect(
            add_query_arg(
                [
                    'notice'      => 'imported',
                    'notice_type' => 'success',
                    'imported'    => $imported,
                    'skipped'     => $skipped,
                ],
                $redirect
            )
        );
        die();
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
            'created'              => __( 'Contact created successfully.', 'myrock-mail-engine' ),
            'updated'              => __( 'Contact updated successfully.', 'myrock-mail-engine' ),
            'deleted'              => __( 'Contact deleted successfully.', 'myrock-mail-engine' ),
            'save_failed'          => __( 'Failed to save contact. Please try again.', 'myrock-mail-engine' ),
            'delete_failed'        => __( 'Failed to delete contact. Please try again.', 'myrock-mail-engine' ),
            'invalid_email'        => __( 'A valid email address is required.', 'myrock-mail-engine' ),
            'import_no_file'       => __( 'No file uploaded or upload error occurred.', 'myrock-mail-engine' ),
            'import_invalid_type'  => __( 'Please upload a valid CSV file.', 'myrock-mail-engine' ),
            'import_upload_failed' => __( 'Failed to process the uploaded file.', 'myrock-mail-engine' ),
            'import_failed'        => __( 'Import failed. Please check the file format and try again.', 'myrock-mail-engine' ),
        ];

        if ( 'imported' === $notice_key ) {
            $imported = isset( $_GET['imported'] ) ? (int) $_GET['imported'] : 0;
            $skipped  = isset( $_GET['skipped'] ) ? (int) $_GET['skipped'] : 0;

            return [
                'type'    => 'success',
                'message' => sprintf(
                    /* translators: 1: imported count, 2: skipped count */
                    __( 'Import complete: %1$d imported, %2$d skipped.', 'myrock-mail-engine' ),
                    $imported,
                    $skipped
                ),
            ];
        }

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
