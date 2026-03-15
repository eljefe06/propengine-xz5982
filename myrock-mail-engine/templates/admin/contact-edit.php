<?php
/**
 * Admin template: Create / Edit a Contact.
 *
 * Available variables (set by ContactsPage before including this template):
 *   @var array|null $contact   Contact row (null when creating a new one).
 *   @var array      $lists     All available mailing lists.
 *   @var array      $tags      All available tags.
 *   @var array      $contact_list_ids  IDs of lists the contact belongs to.
 *   @var array      $contact_tag_ids   IDs of tags assigned to the contact.
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;

$is_new   = empty( $contact );
$base_url = admin_url( 'admin.php?page=mrme-contacts' );

// Helper: pull a field from the contact (array or object).
$field = static function ( string $key, $default = '' ) use ( $contact ) {
	if ( ! $contact ) {
		return $default;
	}
	return is_array( $contact ) ? ( $contact[ $key ] ?? $default ) : ( $contact->$key ?? $default );
};

$contact_id = $field( 'id', 0 );

$contact_list_ids = $contact_list_ids ?? [];
$contact_tag_ids  = $contact_tag_ids  ?? [];

// Try to decode meta if stored as JSON string.
$raw_meta = $field( 'meta', [] );
if ( is_string( $raw_meta ) ) {
	$decoded = json_decode( $raw_meta, true );
	$raw_meta = is_array( $decoded ) ? $decoded : [];
}
$meta_json = ! empty( $raw_meta ) ? wp_json_encode( $raw_meta, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE ) : '';

$status_options = [
	'subscribed'   => __( 'Subscribed', 'myrock-mail-engine' ),
	'unsubscribed' => __( 'Unsubscribed', 'myrock-mail-engine' ),
	'pending'      => __( 'Pending', 'myrock-mail-engine' ),
	'bounced'      => __( 'Bounced', 'myrock-mail-engine' ),
];

$source_options = [
	'form'    => __( 'Form', 'myrock-mail-engine' ),
	'import'  => __( 'Import', 'myrock-mail-engine' ),
	'api'     => __( 'API', 'myrock-mail-engine' ),
	'manual'  => __( 'Manual', 'myrock-mail-engine' ),
];
?>
<div class="wrap mrme-wrap">

	<h1 class="mrme-page-title">
		<?php echo $is_new
			? esc_html__( 'Add Contact', 'myrock-mail-engine' )
			: esc_html__( 'Edit Contact', 'myrock-mail-engine' );
		?>
	</h1>

	<form
		method="POST"
		action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>"
		class="mrme-edit-form"
		novalidate
	>
		<?php wp_nonce_field( 'mrme_save_contact_' . (int) $contact_id, '_wpnonce' ); ?>
		<input type="hidden" name="action" value="mrme_save_contact">
		<?php if ( ! $is_new ) : ?>
		<input type="hidden" name="contact_id" value="<?php echo esc_attr( (int) $contact_id ); ?>">
		<?php endif; ?>

		<div class="mrme-edit-form__layout">

			<!-- ============================================================ -->
			<!-- Left column: contact details                                 -->
			<!-- ============================================================ -->
			<div class="mrme-edit-form__main">

				<!-- Core fields -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Contact Details', 'myrock-mail-engine' ); ?></h2>

					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-email">
							<?php esc_html_e( 'Email Address', 'myrock-mail-engine' ); ?> <span aria-hidden="true">*</span>
						</label>
						<input
							type="email"
							id="mrme-email"
							name="email"
							class="mrme-form-section__input regular-text"
							value="<?php echo esc_attr( $field( 'email' ) ); ?>"
							required
							<?php if ( ! $is_new ) : ?>readonly<?php endif; ?>
						>
						<?php if ( ! $is_new ) : ?>
						<p class="description"><?php esc_html_e( 'Email cannot be changed after creation.', 'myrock-mail-engine' ); ?></p>
						<?php endif; ?>
					</div>

					<div class="mrme-form-section__row mrme-form-section__row--cols">
						<div>
							<label class="mrme-form-section__label" for="mrme-first-name">
								<?php esc_html_e( 'First Name', 'myrock-mail-engine' ); ?>
							</label>
							<input
								type="text"
								id="mrme-first-name"
								name="first_name"
								class="mrme-form-section__input regular-text"
								value="<?php echo esc_attr( $field( 'first_name' ) ); ?>"
							>
						</div>
						<div>
							<label class="mrme-form-section__label" for="mrme-last-name">
								<?php esc_html_e( 'Last Name', 'myrock-mail-engine' ); ?>
							</label>
							<input
								type="text"
								id="mrme-last-name"
								name="last_name"
								class="mrme-form-section__input regular-text"
								value="<?php echo esc_attr( $field( 'last_name' ) ); ?>"
							>
						</div>
					</div>

					<div class="mrme-form-section__row mrme-form-section__row--cols">
						<div>
							<label class="mrme-form-section__label" for="mrme-phone">
								<?php esc_html_e( 'Phone', 'myrock-mail-engine' ); ?>
							</label>
							<input
								type="tel"
								id="mrme-phone"
								name="phone"
								class="mrme-form-section__input regular-text"
								value="<?php echo esc_attr( $field( 'phone' ) ); ?>"
							>
						</div>
						<div>
							<label class="mrme-form-section__label" for="mrme-company">
								<?php esc_html_e( 'Company', 'myrock-mail-engine' ); ?>
							</label>
							<input
								type="text"
								id="mrme-company"
								name="company"
								class="mrme-form-section__input regular-text"
								value="<?php echo esc_attr( $field( 'company' ) ); ?>"
							>
						</div>
					</div>

					<div class="mrme-form-section__row mrme-form-section__row--cols">
						<div>
							<label class="mrme-form-section__label" for="mrme-status">
								<?php esc_html_e( 'Status', 'myrock-mail-engine' ); ?>
							</label>
							<select id="mrme-status" name="status" class="mrme-form-section__select">
								<?php foreach ( $status_options as $val => $label ) : ?>
								<option value="<?php echo esc_attr( $val ); ?>" <?php selected( $field( 'status', 'subscribed' ), $val ); ?>>
									<?php echo esc_html( $label ); ?>
								</option>
								<?php endforeach; ?>
							</select>
						</div>
						<div>
							<label class="mrme-form-section__label" for="mrme-source">
								<?php esc_html_e( 'Source', 'myrock-mail-engine' ); ?>
							</label>
							<select id="mrme-source" name="source" class="mrme-form-section__select">
								<?php foreach ( $source_options as $val => $label ) : ?>
								<option value="<?php echo esc_attr( $val ); ?>" <?php selected( $field( 'source', 'manual' ), $val ); ?>>
									<?php echo esc_html( $label ); ?>
								</option>
								<?php endforeach; ?>
							</select>
						</div>
					</div>

					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-notes">
							<?php esc_html_e( 'Notes', 'myrock-mail-engine' ); ?>
						</label>
						<textarea
							id="mrme-notes"
							name="notes"
							class="mrme-form-section__textarea large-text"
							rows="4"
						><?php echo esc_textarea( $field( 'notes' ) ); ?></textarea>
					</div>

				</div><!-- /.mrme-form-section--card -->

				<!-- Custom meta -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Custom Meta (JSON)', 'myrock-mail-engine' ); ?></h2>
					<p class="description"><?php esc_html_e( 'Store additional key/value data as a JSON object. Example: {"custom_field": "value"}.', 'myrock-mail-engine' ); ?></p>
					<div class="mrme-form-section__row">
						<textarea
							id="mrme-meta"
							name="meta"
							class="mrme-form-section__textarea mrme-meta-textarea large-text code"
							rows="6"
							placeholder="{}"
						><?php echo esc_textarea( $meta_json ); ?></textarea>
						<span id="mrme-meta-error" class="mrme-field-error" style="display:none;">
							<?php esc_html_e( 'Invalid JSON. Please fix before saving.', 'myrock-mail-engine' ); ?>
						</span>
					</div>
				</div>

			</div><!-- /.mrme-edit-form__main -->

			<!-- ============================================================ -->
			<!-- Right column: lists, tags, actions                          -->
			<!-- ============================================================ -->
			<div class="mrme-edit-form__sidebar">

				<!-- Submit box -->
				<div class="mrme-form-section mrme-form-section--card mrme-submit-box">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Save', 'myrock-mail-engine' ); ?></h2>
					<div class="mrme-submit-box__actions">
						<button type="submit" class="button button-primary button-large">
							<?php echo $is_new
								? esc_html__( 'Add Contact', 'myrock-mail-engine' )
								: esc_html__( 'Update Contact', 'myrock-mail-engine' );
							?>
						</button>
						<a href="<?php echo esc_url( $base_url ); ?>" class="button button-large">
							<?php esc_html_e( 'Cancel', 'myrock-mail-engine' ); ?>
						</a>
					</div>
				</div>

				<!-- Lists -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Mailing Lists', 'myrock-mail-engine' ); ?></h2>
					<?php if ( ! empty( $lists ) ) : ?>
					<div class="mrme-checkbox-grid">
						<?php foreach ( $lists as $list ) :
							$l_id   = is_array( $list ) ? $list['id']   : $list->id;
							$l_name = is_array( $list ) ? $list['name'] : $list->name;
						?>
						<label class="mrme-checkbox-grid__item">
							<input
								type="checkbox"
								name="list_ids[]"
								value="<?php echo esc_attr( (int) $l_id ); ?>"
								<?php checked( in_array( (int) $l_id, array_map( 'intval', $contact_list_ids ), true ) ); ?>
							>
							<?php echo esc_html( $l_name ); ?>
						</label>
						<?php endforeach; ?>
					</div>
					<?php else : ?>
					<p class="mrme-muted"><?php esc_html_e( 'No lists available.', 'myrock-mail-engine' ); ?></p>
					<?php endif; ?>
				</div>

				<!-- Tags -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Tags', 'myrock-mail-engine' ); ?></h2>
					<?php if ( ! empty( $tags ) ) : ?>
					<div class="mrme-checkbox-grid">
						<?php foreach ( $tags as $tag ) :
							$t_id   = is_array( $tag ) ? $tag['id']   : $tag->id;
							$t_name = is_array( $tag ) ? $tag['name'] : $tag->name;
						?>
						<label class="mrme-checkbox-grid__item">
							<input
								type="checkbox"
								name="tag_ids[]"
								value="<?php echo esc_attr( (int) $t_id ); ?>"
								<?php checked( in_array( (int) $t_id, array_map( 'intval', $contact_tag_ids ), true ) ); ?>
							>
							<?php echo esc_html( $t_name ); ?>
						</label>
						<?php endforeach; ?>
					</div>
					<?php else : ?>
					<p class="mrme-muted"><?php esc_html_e( 'No tags available.', 'myrock-mail-engine' ); ?></p>
					<?php endif; ?>
				</div>

				<!-- Metadata (read-only) -->
				<?php if ( ! $is_new ) : ?>
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Record Info', 'myrock-mail-engine' ); ?></h2>
					<dl class="mrme-dl">
						<dt><?php esc_html_e( 'Created', 'myrock-mail-engine' ); ?></dt>
						<dd>
							<?php
							$created_at = $field( 'created_at', '' );
							echo esc_html( $created_at ? wp_date( get_option( 'date_format' ) . ' ' . get_option( 'time_format' ), strtotime( $created_at ) ) : '—' );
							?>
						</dd>
						<dt><?php esc_html_e( 'Last Updated', 'myrock-mail-engine' ); ?></dt>
						<dd>
							<?php
							$updated_at = $field( 'updated_at', '' );
							echo esc_html( $updated_at ? wp_date( get_option( 'date_format' ) . ' ' . get_option( 'time_format' ), strtotime( $updated_at ) ) : '—' );
							?>
						</dd>
						<dt><?php esc_html_e( 'IP Address', 'myrock-mail-engine' ); ?></dt>
						<dd><?php echo esc_html( $field( 'ip_address', '—' ) ); ?></dd>
					</dl>
				</div>
				<?php endif; ?>

			</div><!-- /.mrme-edit-form__sidebar -->

		</div><!-- /.mrme-edit-form__layout -->

	</form>

</div><!-- /.wrap.mrme-wrap -->
