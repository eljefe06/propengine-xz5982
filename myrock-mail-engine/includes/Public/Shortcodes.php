<?php
namespace MyRock\MailEngine\Public;
/**
 * Shortcodes — registers and renders the [mrme_form] shortcode.
 *
 * @package MyRock\MailEngine\Public
 */

defined( 'ABSPATH' ) || exit;


use MyRock\MailEngine\Models\Form;

/**
 * Class Shortcodes
 *
 * Provides the [mrme_form id="X"] shortcode that renders a subscription form.
 */
class Shortcodes {

	/**
	 * Register all plugin shortcodes.
	 *
	 * @return void
	 */
	public function register_shortcodes(): void {
		add_shortcode( 'mrme_form', [ self::class, 'render_form' ] );
	}

	/**
	 * Enqueue public-facing assets (CSS + JS).
	 *
	 * @return void
	 */
	public function enqueue_assets(): void {
		wp_enqueue_style(
			'mrme-public',
			MRME_URL . 'assets/css/public.css',
			[],
			MRME_VERSION
		);

		wp_enqueue_script(
			'mrme-public',
			MRME_URL . 'assets/js/public.js',
			[ 'jquery' ],
			MRME_VERSION,
			true
		);

		// Pass AJAX URL so the public JS can use AJAX mode when data-ajax="true".
		wp_localize_script(
			'mrme-public',
			'mrmePublic',
			[
				'ajaxurl'  => admin_url( 'admin-ajax.php' ),
				'posturl'  => admin_url( 'admin-post.php' ),
				'i18n'     => [
					'required'     => __( 'Este campo es obligatorio.', 'myrock-mail-engine' ),
					'invalid_email'=> __( 'Por favor introduce un email válido.', 'myrock-mail-engine' ),
					'success'      => __( '¡Gracias por suscribirte!', 'myrock-mail-engine' ),
					'error'        => __( 'Ha ocurrido un error. Inténtalo de nuevo.', 'myrock-mail-engine' ),
				],
			]
		);
	}

	/**
	 * Render the subscription form HTML.
	 *
	 * Shortcode attributes:
	 *   - id    (required) — Form post/record ID.
	 *   - class (optional) — Additional CSS class(es) added to the <form> element.
	 *
	 * @param array $atts Shortcode attributes.
	 * @return string Rendered form HTML, or empty string on failure.
	 */
	public static function render_form( array $atts ): string {

		$atts = shortcode_atts(
			[
				'id'    => 0,
				'class' => '',
			],
			$atts,
			'mrme_form'
		);

		$form_id = (int) $atts['id'];

		if ( $form_id <= 0 ) {
			return '';
		}

		// Retrieve the form record.
		$form = Form::find( $form_id );

		if ( ! $form ) {
			return '';
		}

		// Accept both array and object shapes for status.
		$status = is_array( $form ) ? ( $form['status'] ?? '' ) : ( $form->status ?? '' );

		if ( 'active' !== $status ) {
			return '';
		}

		// Retrieve dynamic field definitions and associated list IDs.
		$fields  = Form::get_fields( $form_id );
		$list_ids = Form::get_list_ids( $form_id );

		if ( ! is_array( $fields ) ) {
			$fields = [];
		}

		if ( ! is_array( $list_ids ) ) {
			$list_ids = [];
		}

		// Build extra CSS classes.
		$extra_class = sanitize_html_class( $atts['class'] );
		$form_classes = trim( 'mrme-form ' . $extra_class );

		// Determine success/error notice from query string (after redirect-back).
		$notice_html = '';
		// phpcs:ignore WordPress.Security.NonceVerification.Recommended
		if ( isset( $_GET['mrme'] ) ) {
			// phpcs:ignore WordPress.Security.NonceVerification.Recommended
			$mrme_param = sanitize_key( $_GET['mrme'] );
			if ( 'subscribed' === $mrme_param ) {
				$notice_html = '<div class="mrme-form__notice mrme-form--success" role="alert">'
					. esc_html__( '¡Gracias! Te has suscrito correctamente.', 'myrock-mail-engine' )
					. '</div>';
			} elseif ( 'error' === $mrme_param ) {
				$notice_html = '<div class="mrme-form__notice mrme-form--error" role="alert">'
					. esc_html__( 'Ha ocurrido un error. Por favor, inténtalo de nuevo.', 'myrock-mail-engine' )
					. '</div>';
			}
		}

		// Start output buffering so we can return a string.
		ob_start();
		?>
		<div class="mrme-form-wrapper">
			<?php echo $notice_html; // Already escaped above. ?>
			<form
				action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>"
				method="POST"
				class="<?php echo esc_attr( $form_classes ); ?>"
				data-form-id="<?php echo esc_attr( $form_id ); ?>"
			>
				<?php // Hidden action field required by admin-post.php. ?>
				<input type="hidden" name="action" value="mrme_subscribe">

				<?php // Form identifier. ?>
				<input type="hidden" name="form_id" value="<?php echo esc_attr( $form_id ); ?>">

				<?php
				// Nonce for CSRF protection, keyed to this specific form.
				// We output a hidden field manually so we control the field name.
				echo '<input type="hidden" name="_wpnonce" value="'
					. esc_attr( wp_create_nonce( 'mrme_subscribe_' . $form_id ) )
					. '">';
				?>

				<?php
				// List IDs — hidden fields so FormHandler can assign the contact.
				foreach ( $list_ids as $lid ) {
					echo '<input type="hidden" name="list_ids[]" value="' . esc_attr( (int) $lid ) . '">';
				}
				?>

				<?php
				// Render dynamic fields.
				foreach ( $fields as $field ) :
					$field_name     = is_array( $field ) ? ( $field['name'] ?? '' )     : ( $field->name ?? '' );
					$field_label    = is_array( $field ) ? ( $field['label'] ?? '' )    : ( $field->label ?? '' );
					$field_type     = is_array( $field ) ? ( $field['type'] ?? 'text' ) : ( $field->type ?? 'text' );
					$field_required = is_array( $field ) ? ! empty( $field['required'] ) : ! empty( $field->required );

					// Only allow safe input types.
					$allowed_types  = [ 'text', 'email', 'tel' ];
					if ( ! in_array( $field_type, $allowed_types, true ) ) {
						$field_type = 'text';
					}

					if ( empty( $field_name ) ) {
						continue;
					}

					$input_id = 'mrme_field_' . $form_id . '_' . esc_attr( $field_name );
				?>
				<div class="mrme-form__field">
					<label
						class="mrme-form__label"
						for="<?php echo esc_attr( $input_id ); ?>"
					>
						<?php echo esc_html( $field_label ?: $field_name ); ?>
						<?php if ( $field_required ) : ?>
							<span class="mrme-form__required" aria-hidden="true"> *</span>
						<?php endif; ?>
					</label>
					<input
						type="<?php echo esc_attr( $field_type ); ?>"
						id="<?php echo esc_attr( $input_id ); ?>"
						name="<?php echo esc_attr( $field_name ); ?>"
						class="mrme-form__input"
						<?php if ( $field_required ) : ?>
							required
							aria-required="true"
						<?php endif; ?>
					>
				</div>
				<?php endforeach; ?>

				<div class="mrme-form__field mrme-form__field--submit">
					<button type="submit" class="mrme-form__submit">
						<?php esc_html_e( 'Suscribirme', 'myrock-mail-engine' ); ?>
					</button>
				</div>

			</form>
		</div>
		<?php
		return ob_get_clean();
	}
}
