<?php
namespace MyRock\MailEngine\Public;
/**
 * FormHandler — processes subscription, unsubscribe, and optin-confirmation requests.
 *
 * @package MyRock\MailEngine\Public
 */

defined( 'ABSPATH' ) || exit;


use MyRock\MailEngine\Models\Form;
use MyRock\MailEngine\Models\Contact;
use MyRock\MailEngine\Services\ContactService;
use MyRock\MailEngine\Services\MailManager;

/**
 * Class FormHandler
 *
 * Hooks into WordPress to handle:
 *  - Front-end form submissions  (admin-post: mrme_subscribe)
 *  - Unsubscribe token links     (init: ?mrme_action=unsubscribe&token=…)
 *  - Double opt-in confirmations (init: ?mrme_action=confirm_optin&token=…)
 */
class FormHandler {

	/**
	 * Process a subscription form POST.
	 *
	 * Hooked to:
	 *   admin_post_mrme_subscribe         (for logged-in users)
	 *   admin_post_nopriv_mrme_subscribe  (for guests)
	 *
	 * @return void
	 */
	public function handle_subscribe(): void {

		// ------------------------------------------------------------------ //
		// 1. Basic sanity: form_id must be present.
		// ------------------------------------------------------------------ //
		$form_id = isset( $_POST['form_id'] ) ? (int) $_POST['form_id'] : 0;

		if ( $form_id <= 0 ) {
			$this->redirect_back( 'error' );
		}

		// ------------------------------------------------------------------ //
		// 2. Nonce verification.
		// ------------------------------------------------------------------ //
		$nonce = isset( $_POST['_wpnonce'] ) ? sanitize_text_field( wp_unslash( $_POST['_wpnonce'] ) ) : '';

		if ( ! wp_verify_nonce( $nonce, 'mrme_subscribe_' . $form_id ) ) {
			$this->redirect_back( 'error' );
		}

		// ------------------------------------------------------------------ //
		// 3. Load and validate the form.
		// ------------------------------------------------------------------ //
		$form = Form::find( $form_id );

		if ( ! $form ) {
			$this->redirect_back( 'error' );
		}

		$form_status = is_array( $form ) ? ( $form['status'] ?? '' ) : ( $form->status ?? '' );

		if ( 'active' !== $form_status ) {
			$this->redirect_back( 'error' );
		}

		// ------------------------------------------------------------------ //
		// 4. Collect field values from POST.
		// ------------------------------------------------------------------ //
		$fields = Form::get_fields( $form_id );

		if ( ! is_array( $fields ) ) {
			$fields = [];
		}

		$contact_data = [];

		// Map of well-known field names to the Contact model's column names.
		$known_fields = [
			'email'      => 'email',
			'first_name' => 'first_name',
			'last_name'  => 'last_name',
			'phone'      => 'phone',
			'company'    => 'company',
		];

		foreach ( $fields as $field ) {
			$field_name = is_array( $field ) ? ( $field['name'] ?? '' ) : ( $field->name ?? '' );

			if ( empty( $field_name ) ) {
				continue;
			}

			// phpcs:ignore WordPress.Security.ValidatedSanitizedInput.InputNotValidated
			$raw_value = isset( $_POST[ $field_name ] ) ? sanitize_text_field( wp_unslash( $_POST[ $field_name ] ) ) : '';

			if ( isset( $known_fields[ $field_name ] ) ) {
				$contact_data[ $known_fields[ $field_name ] ] = $raw_value;
			} else {
				// Store unknown fields in a meta bag.
				$contact_data['meta'][ $field_name ] = $raw_value;
			}
		}

		// ------------------------------------------------------------------ //
		// 5. Validate email.
		// ------------------------------------------------------------------ //
		$email = isset( $contact_data['email'] ) ? sanitize_email( $contact_data['email'] ) : '';

		if ( ! is_email( $email ) ) {
			$this->redirect_back( 'error' );
		}

		$contact_data['email'] = $email;

		// ------------------------------------------------------------------ //
		// 6. Determine double opt-in setting.
		// ------------------------------------------------------------------ //
		$double_optin = is_array( $form )
			? ! empty( $form['double_optin'] )
			: ! empty( $form->double_optin ?? false );

		// ------------------------------------------------------------------ //
		// 7. Build contact payload.
		// ------------------------------------------------------------------ //
		$contact_data['status']     = $double_optin ? 'pending' : 'subscribed';
		$contact_data['source']     = 'form';
		$contact_data['ip_address'] = sanitize_text_field(
			wp_unslash( $_SERVER['REMOTE_ADDR'] ?? '' )
		);

		$list_ids = Form::get_list_ids( $form_id );
		$tag_ids  = Form::get_tag_ids( $form_id );

		if ( ! is_array( $list_ids ) ) {
			$list_ids = [];
		}

		if ( ! is_array( $tag_ids ) ) {
			$tag_ids = [];
		}

		$contact_data['list_ids'] = $list_ids;
		$contact_data['tag_ids']  = $tag_ids;

		// ------------------------------------------------------------------ //
		// 8. Create or update the contact.
		// ------------------------------------------------------------------ //
		$contact_id = ContactService::create_or_update( $contact_data );

		if ( ! $contact_id ) {
			$this->redirect_back( 'error' );
		}

		// ------------------------------------------------------------------ //
		// 9. Send double opt-in confirmation email if required.
		// ------------------------------------------------------------------ //
		if ( $double_optin ) {
			$confirmation_token = ContactService::generate_token( $contact_id, 'confirm_optin' );
			$confirm_url        = add_query_arg(
				[
					'mrme_action' => 'confirm_optin',
					'token'       => $confirmation_token,
				],
				home_url( '/' )
			);

			$from_name  = get_option( 'mrme_from_name', get_bloginfo( 'name' ) );
			$from_email = get_option( 'mrme_from_email', get_option( 'admin_email' ) );

			$subject = sprintf(
				/* translators: %s: site name */
				__( 'Confirma tu suscripción a %s', 'myrock-mail-engine' ),
				get_bloginfo( 'name' )
			);

			$body = sprintf(
				'<p>%s</p><p><a href="%s">%s</a></p>',
				esc_html__( 'Por favor, confirma tu suscripción haciendo clic en el siguiente enlace:', 'myrock-mail-engine' ),
				esc_url( $confirm_url ),
				esc_html__( 'Confirmar suscripción', 'myrock-mail-engine' )
			);

			MailManager::send_raw(
				$email,
				$subject,
				$body,
				[
					'from_name'  => $from_name,
					'from_email' => $from_email,
				]
			);
		}

		// ------------------------------------------------------------------ //
		// 10. Redirect.
		// ------------------------------------------------------------------ //
		$redirect_url = is_array( $form )
			? ( $form['redirect_url'] ?? '' )
			: ( $form->redirect_url ?? '' );

		if ( ! empty( $redirect_url ) ) {
			$redirect_url = esc_url_raw( $redirect_url );
			wp_safe_redirect( add_query_arg( 'mrme', 'subscribed', $redirect_url ) );
			exit;
		}

		$this->redirect_back( 'subscribed' );
	}

	/**
	 * Process an unsubscribe token link.
	 *
	 * Hooked to: init
	 * Triggered by: ?mrme_action=unsubscribe&token=…
	 *
	 * @return void
	 */
	public function handle_unsubscribe(): void {

		// phpcs:disable WordPress.Security.NonceVerification.Recommended
		if (
			! isset( $_GET['mrme_action'] ) ||
			'unsubscribe' !== sanitize_key( $_GET['mrme_action'] ) ||
			! isset( $_GET['token'] )
		) {
			return;
		}

		$token = sanitize_text_field( wp_unslash( $_GET['token'] ) );
		// phpcs:enable WordPress.Security.NonceVerification.Recommended

		if ( empty( $token ) ) {
			wp_safe_redirect( add_query_arg( 'mrme', 'error', home_url( '/' ) ) );
			exit;
		}

		$result = ContactService::unsubscribe_by_token( $token );

		if ( $result ) {
			// Set a short-lived transient so themes/plugins can display a notice.
			set_transient( 'mrme_notice_unsubscribed_' . md5( $token ), true, 60 );
		}

		wp_safe_redirect(
			add_query_arg( 'mrme', $result ? 'unsubscribed' : 'error', home_url( '/' ) )
		);
		exit;
	}

	/**
	 * Process a double opt-in confirmation token link.
	 *
	 * Hooked to: init
	 * Triggered by: ?mrme_action=confirm_optin&token=…
	 *
	 * @return void
	 */
	public function handle_confirm_optin(): void {

		// phpcs:disable WordPress.Security.NonceVerification.Recommended
		if (
			! isset( $_GET['mrme_action'] ) ||
			'confirm_optin' !== sanitize_key( $_GET['mrme_action'] ) ||
			! isset( $_GET['token'] )
		) {
			return;
		}

		$token = sanitize_text_field( wp_unslash( $_GET['token'] ) );
		// phpcs:enable WordPress.Security.NonceVerification.Recommended

		if ( empty( $token ) ) {
			wp_safe_redirect( add_query_arg( 'mrme', 'error', home_url( '/' ) ) );
			exit;
		}

		$result = ContactService::confirm_optin_by_token( $token );

		wp_safe_redirect(
			add_query_arg( 'mrme', $result ? 'confirmed' : 'error', home_url( '/' ) )
		);
		exit;
	}

	// ------------------------------------------------------------------ //
	// Private helpers.
	// ------------------------------------------------------------------ //

	/**
	 * Redirect back to the referrer (or home) with a status query arg.
	 *
	 * @param string $status mrme query-string value ('subscribed', 'error', etc.).
	 * @return void  Always calls exit.
	 */
	private function redirect_back( string $status ): void {
		$referer = wp_get_referer();

		if ( ! $referer ) {
			$referer = home_url( '/' );
		}

		// Remove any pre-existing mrme param to avoid duplication.
		$referer = remove_query_arg( 'mrme', $referer );

		wp_safe_redirect( add_query_arg( 'mrme', $status, $referer ) );
		exit;
	}
}
