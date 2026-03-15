<?php
/**
 * Admin template: Create / Edit a Campaign.
 *
 * Available variables (set by CampaignsPage before including this template):
 *   @var array|null $campaign     Campaign row (null for new).
 *   @var array      $lists        All available mailing lists.
 *   @var array      $tags         All available tags.
 *   @var array      $campaign_list_ids  List IDs already assigned.
 *   @var array      $campaign_tag_ids   Tag IDs already assigned.
 *   @var array|null $stats        Campaign send stats (or null if not sent).
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;

$is_new   = empty( $campaign );
$base_url = admin_url( 'admin.php?page=mrme-campaigns' );

$field = static function ( string $key, $default = '' ) use ( $campaign ) {
	if ( ! $campaign ) {
		return $default;
	}
	return is_array( $campaign ) ? ( $campaign[ $key ] ?? $default ) : ( $campaign->$key ?? $default );
};

$campaign_id = (int) $field( 'id', 0 );
$c_status    = $field( 'status', 'draft' );

$campaign_list_ids = $campaign_list_ids ?? [];
$campaign_tag_ids  = $campaign_tag_ids  ?? [];
$stats             = $stats ?? null;

// Scheduled date formatted for datetime-local input (YYYY-MM-DDTHH:MM).
$scheduled_at = $field( 'scheduled_at', '' );
$scheduled_dt = '';
if ( $scheduled_at ) {
	$ts           = strtotime( $scheduled_at );
	$scheduled_dt = $ts ? gmdate( 'Y-m-d\TH:i', $ts ) : '';
}

// Default sender settings from plugin options (fallback for new campaigns).
$default_from_name  = get_option( 'mrme_from_name', get_bloginfo( 'name' ) );
$default_from_email = get_option( 'mrme_from_email', get_option( 'admin_email' ) );
$default_reply_to   = get_option( 'mrme_reply_to', $default_from_email );
?>
<div class="wrap mrme-wrap">

	<h1 class="mrme-page-title">
		<?php echo $is_new
			? esc_html__( 'Create Campaign', 'myrock-mail-engine' )
			: esc_html__( 'Edit Campaign', 'myrock-mail-engine' );
		?>
		<?php if ( ! $is_new ) : ?>
		<span class="mrme-badge mrme-badge--<?php echo esc_attr( $c_status ); ?> mrme-badge--title">
			<?php echo esc_html( ucfirst( $c_status ) ); ?>
		</span>
		<?php endif; ?>
	</h1>

	<form
		method="POST"
		action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>"
		class="mrme-edit-form"
		id="mrme-campaign-form"
		novalidate
	>
		<?php wp_nonce_field( 'mrme_save_campaign_' . $campaign_id, '_wpnonce' ); ?>
		<input type="hidden" name="action" value="mrme_save_campaign">
		<?php if ( ! $is_new ) : ?>
		<input type="hidden" name="campaign_id" value="<?php echo esc_attr( $campaign_id ); ?>">
		<?php endif; ?>

		<div class="mrme-edit-form__layout">

			<!-- ============================================================ -->
			<!-- Main column                                                   -->
			<!-- ============================================================ -->
			<div class="mrme-edit-form__main">

				<!-- Basic fields -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Campaign Details', 'myrock-mail-engine' ); ?></h2>

					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-campaign-title">
							<?php esc_html_e( 'Campaign Title', 'myrock-mail-engine' ); ?> <span aria-hidden="true">*</span>
						</label>
						<input
							type="text"
							id="mrme-campaign-title"
							name="title"
							class="mrme-form-section__input large-text"
							value="<?php echo esc_attr( $field( 'title' ) ); ?>"
							required
							placeholder="<?php esc_attr_e( 'Internal name for this campaign', 'myrock-mail-engine' ); ?>"
						>
					</div>

					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-subject">
							<?php esc_html_e( 'Email Subject', 'myrock-mail-engine' ); ?> <span aria-hidden="true">*</span>
						</label>
						<input
							type="text"
							id="mrme-subject"
							name="subject"
							class="mrme-form-section__input large-text"
							value="<?php echo esc_attr( $field( 'subject' ) ); ?>"
							required
							placeholder="<?php esc_attr_e( 'The subject line your subscribers will see', 'myrock-mail-engine' ); ?>"
						>
					</div>

					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-preheader">
							<?php esc_html_e( 'Preheader Text', 'myrock-mail-engine' ); ?>
						</label>
						<input
							type="text"
							id="mrme-preheader"
							name="preheader"
							class="mrme-form-section__input large-text"
							value="<?php echo esc_attr( $field( 'preheader' ) ); ?>"
							placeholder="<?php esc_attr_e( 'Short preview text shown next to the subject', 'myrock-mail-engine' ); ?>"
						>
					</div>
				</div>

				<!-- Sender -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Sender Information', 'myrock-mail-engine' ); ?></h2>

					<div class="mrme-form-section__row mrme-form-section__row--cols">
						<div>
							<label class="mrme-form-section__label" for="mrme-from-name">
								<?php esc_html_e( 'From Name', 'myrock-mail-engine' ); ?>
							</label>
							<input
								type="text"
								id="mrme-from-name"
								name="from_name"
								class="mrme-form-section__input regular-text"
								value="<?php echo esc_attr( $field( 'from_name', $default_from_name ) ); ?>"
							>
						</div>
						<div>
							<label class="mrme-form-section__label" for="mrme-from-email">
								<?php esc_html_e( 'From Email', 'myrock-mail-engine' ); ?> <span aria-hidden="true">*</span>
							</label>
							<input
								type="email"
								id="mrme-from-email"
								name="from_email"
								class="mrme-form-section__input regular-text"
								value="<?php echo esc_attr( $field( 'from_email', $default_from_email ) ); ?>"
								required
							>
						</div>
					</div>

					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-reply-to">
							<?php esc_html_e( 'Reply-To', 'myrock-mail-engine' ); ?>
						</label>
						<input
							type="email"
							id="mrme-reply-to"
							name="reply_to"
							class="mrme-form-section__input regular-text"
							value="<?php echo esc_attr( $field( 'reply_to', $default_reply_to ) ); ?>"
							placeholder="<?php echo esc_attr( $default_from_email ); ?>"
						>
					</div>
				</div>

				<!-- Content editor -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Email Content', 'myrock-mail-engine' ); ?></h2>

					<?php
					$content_html = $field( 'content_html', '' );
					$editor_id    = 'mrme_content_html';

					if ( function_exists( 'wp_editor' ) ) :
						wp_editor(
							$content_html,
							$editor_id,
							[
								'textarea_name' => 'content_html',
								'textarea_rows' => 20,
								'media_buttons' => true,
								'tinymce'       => [
									'toolbar1' => 'formatselect,bold,italic,underline,blockquote,bullist,numlist,alignleft,aligncenter,alignright,link,unlink,image,undo,redo',
									'toolbar2' => '',
								],
							]
						);
					else : ?>
					<div class="mrme-form-section__row">
						<textarea
							id="<?php echo esc_attr( $editor_id ); ?>"
							name="content_html"
							class="mrme-form-section__textarea large-text code"
							rows="20"
						><?php echo esc_textarea( $content_html ); ?></textarea>
						<p class="description"><?php esc_html_e( 'You may use HTML. Use {{first_name}}, {{last_name}}, {{email}}, {{unsubscribe_url}} as merge tags.', 'myrock-mail-engine' ); ?></p>
					</div>
					<?php endif; ?>

					<p class="description mrme-merge-tags-help">
						<?php esc_html_e( 'Available merge tags:', 'myrock-mail-engine' ); ?>
						<code>{{first_name}}</code>, <code>{{last_name}}</code>, <code>{{email}}</code>,
						<code>{{unsubscribe_url}}</code>, <code>{{site_name}}</code>
					</p>
				</div>

				<!-- Test send -->
				<div class="mrme-form-section mrme-form-section--card" id="mrme-test-send">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Send Test Email', 'myrock-mail-engine' ); ?></h2>
					<div class="mrme-form-section__row mrme-form-section__row--inline">
						<input
							type="email"
							id="mrme-test-email"
							class="mrme-form-section__input regular-text"
							placeholder="<?php esc_attr_e( 'you@example.com', 'myrock-mail-engine' ); ?>"
							<?php if ( $is_new ) : ?>disabled title="<?php esc_attr_e( 'Save the campaign first to enable test send.', 'myrock-mail-engine' ); ?>"<?php endif; ?>
						>
						<button
							type="button"
							id="mrme-send-test-btn"
							class="button"
							data-campaign-id="<?php echo esc_attr( $campaign_id ); ?>"
							data-nonce="<?php echo esc_attr( wp_create_nonce( 'mrme_send_test' ) ); ?>"
							<?php if ( $is_new ) : ?>disabled<?php endif; ?>
						>
							<?php esc_html_e( 'Send Test', 'myrock-mail-engine' ); ?>
						</button>
					</div>
					<div id="mrme-test-send-msg" class="mrme-inline-notice" style="display:none;"></div>
					<?php if ( $is_new ) : ?>
					<p class="description"><?php esc_html_e( 'Save the campaign as a draft first, then you can send a test email.', 'myrock-mail-engine' ); ?></p>
					<?php endif; ?>
				</div>

				<!-- Campaign stats (shown only for sent campaigns) -->
				<?php if ( ! $is_new && 'sent' === $c_status && ! empty( $stats ) ) : ?>
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Campaign Stats', 'myrock-mail-engine' ); ?></h2>
					<div class="mrme-stat-cards mrme-stat-cards--small">
						<div class="mrme-stat-card">
							<div class="mrme-stat-card__value"><?php echo esc_html( number_format_i18n( (int) ( $stats['sent'] ?? 0 ) ) ); ?></div>
							<div class="mrme-stat-card__label"><?php esc_html_e( 'Sent', 'myrock-mail-engine' ); ?></div>
						</div>
						<div class="mrme-stat-card">
							<div class="mrme-stat-card__value"><?php echo esc_html( number_format_i18n( (int) ( $stats['delivered'] ?? 0 ) ) ); ?></div>
							<div class="mrme-stat-card__label"><?php esc_html_e( 'Delivered', 'myrock-mail-engine' ); ?></div>
						</div>
						<div class="mrme-stat-card">
							<div class="mrme-stat-card__value">
								<?php
								$opens      = (int) ( $stats['opens']   ?? 0 );
								$sent       = (int) ( $stats['sent']    ?? 0 );
								$open_rate  = $sent > 0 ? round( $opens / $sent * 100, 1 ) : 0;
								echo esc_html( number_format_i18n( $opens ) . ' (' . $open_rate . '%)' );
								?>
							</div>
							<div class="mrme-stat-card__label"><?php esc_html_e( 'Opens', 'myrock-mail-engine' ); ?></div>
						</div>
						<div class="mrme-stat-card">
							<div class="mrme-stat-card__value">
								<?php
								$clicks      = (int) ( $stats['clicks']  ?? 0 );
								$click_rate  = $sent > 0 ? round( $clicks / $sent * 100, 1 ) : 0;
								echo esc_html( number_format_i18n( $clicks ) . ' (' . $click_rate . '%)' );
								?>
							</div>
							<div class="mrme-stat-card__label"><?php esc_html_e( 'Clicks', 'myrock-mail-engine' ); ?></div>
						</div>
						<div class="mrme-stat-card">
							<div class="mrme-stat-card__value"><?php echo esc_html( number_format_i18n( (int) ( $stats['bounces'] ?? 0 ) ) ); ?></div>
							<div class="mrme-stat-card__label"><?php esc_html_e( 'Bounces', 'myrock-mail-engine' ); ?></div>
						</div>
						<div class="mrme-stat-card">
							<div class="mrme-stat-card__value"><?php echo esc_html( number_format_i18n( (int) ( $stats['unsubscribes'] ?? 0 ) ) ); ?></div>
							<div class="mrme-stat-card__label"><?php esc_html_e( 'Unsubscribes', 'myrock-mail-engine' ); ?></div>
						</div>
					</div>
				</div>
				<?php endif; ?>

			</div><!-- /.mrme-edit-form__main -->

			<!-- ============================================================ -->
			<!-- Sidebar                                                       -->
			<!-- ============================================================ -->
			<div class="mrme-edit-form__sidebar">

				<!-- Publish / actions box -->
				<div class="mrme-form-section mrme-form-section--card mrme-submit-box">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Publish', 'myrock-mail-engine' ); ?></h2>

					<div class="mrme-submit-box__actions">
						<?php if ( ! $is_new && 'sent' === $c_status ) : ?>
						<p class="mrme-muted"><?php esc_html_e( 'This campaign has already been sent.', 'myrock-mail-engine' ); ?></p>
						<?php else : ?>
						<!-- Save Draft -->
						<button type="submit" name="save_action" value="draft" class="button button-large">
							<?php esc_html_e( 'Save Draft', 'myrock-mail-engine' ); ?>
						</button>
						<!-- Send Now -->
						<button
							type="submit"
							name="save_action"
							value="send_now"
							class="button button-primary button-large mrme-action--send"
							data-confirm="<?php esc_attr_e( 'Send this campaign now to all matching recipients?', 'myrock-mail-engine' ); ?>"
						>
							<?php esc_html_e( 'Send Now', 'myrock-mail-engine' ); ?>
						</button>
						<?php endif; ?>
					</div>

					<div class="mrme-submit-box__cancel">
						<a href="<?php echo esc_url( $base_url ); ?>">
							&larr; <?php esc_html_e( 'Back to Campaigns', 'myrock-mail-engine' ); ?>
						</a>
					</div>
				</div>

				<!-- Schedule -->
				<?php if ( $is_new || in_array( $c_status, [ 'draft', 'scheduled' ], true ) ) : ?>
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Schedule', 'myrock-mail-engine' ); ?></h2>
					<div class="mrme-form-section__row">
						<label class="mrme-form-section__label" for="mrme-scheduled-at">
							<?php esc_html_e( 'Send at (leave empty to send immediately)', 'myrock-mail-engine' ); ?>
						</label>
						<input
							type="datetime-local"
							id="mrme-scheduled-at"
							name="scheduled_at"
							class="mrme-form-section__input"
							value="<?php echo esc_attr( $scheduled_dt ); ?>"
						>
					</div>
					<p class="description">
						<?php printf(
							/* translators: %s: WordPress timezone */
							esc_html__( 'Times are in %s timezone.', 'myrock-mail-engine' ),
							esc_html( wp_timezone_string() )
						); ?>
					</p>
				</div>
				<?php endif; ?>

				<!-- Lists -->
				<div class="mrme-form-section mrme-form-section--card">
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Send to Lists', 'myrock-mail-engine' ); ?></h2>
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
								<?php checked( in_array( (int) $l_id, array_map( 'intval', $campaign_list_ids ), true ) ); ?>
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
					<h2 class="mrme-form-section__title"><?php esc_html_e( 'Filter by Tags', 'myrock-mail-engine' ); ?></h2>
					<p class="description"><?php esc_html_e( 'Only contacts with ALL selected tags will receive this campaign.', 'myrock-mail-engine' ); ?></p>
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
								<?php checked( in_array( (int) $t_id, array_map( 'intval', $campaign_tag_ids ), true ) ); ?>
							>
							<?php echo esc_html( $t_name ); ?>
						</label>
						<?php endforeach; ?>
					</div>
					<?php else : ?>
					<p class="mrme-muted"><?php esc_html_e( 'No tags available.', 'myrock-mail-engine' ); ?></p>
					<?php endif; ?>
				</div>

			</div><!-- /.mrme-edit-form__sidebar -->

		</div><!-- /.mrme-edit-form__layout -->

	</form>

</div><!-- /.wrap.mrme-wrap -->
