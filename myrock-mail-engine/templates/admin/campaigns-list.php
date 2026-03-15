<?php
/**
 * Admin template: Campaigns list.
 *
 * Available variables (set by CampaignsPage before including this template):
 *   @var array  $campaigns      Paged array of campaign rows.
 *   @var int    $total          Total count for current filter.
 *   @var int    $paged          Current page number.
 *   @var int    $per_page       Rows per page.
 *   @var string $status_filter  Active status tab value.
 *   @var string $search         Active search string.
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;

$total_pages = $per_page > 0 ? (int) ceil( $total / $per_page ) : 1;
$base_url    = admin_url( 'admin.php?page=mrme-campaigns' );

$status_tabs = [
	''          => __( 'All', 'myrock-mail-engine' ),
	'draft'     => __( 'Draft', 'myrock-mail-engine' ),
	'scheduled' => __( 'Scheduled', 'myrock-mail-engine' ),
	'sending'   => __( 'Sending', 'myrock-mail-engine' ),
	'sent'      => __( 'Sent', 'myrock-mail-engine' ),
];
?>
<div class="wrap mrme-wrap">

	<!-- Page header -->
	<div class="mrme-page-header">
		<h1 class="mrme-page-title"><?php esc_html_e( 'Campaigns', 'myrock-mail-engine' ); ?></h1>
		<div class="mrme-page-header__actions">
			<a href="<?php echo esc_url( add_query_arg( 'action', 'new', $base_url ) ); ?>" class="button button-primary">
				<?php esc_html_e( '+ Create Campaign', 'myrock-mail-engine' ); ?>
			</a>
		</div>
	</div>

	<!-- Filter bar -->
	<div class="mrme-filter-bar">

		<ul class="mrme-status-tabs">
			<?php foreach ( $status_tabs as $tab_key => $tab_label ) :
				$tab_url    = $tab_key ? add_query_arg( 'status', $tab_key, $base_url ) : $base_url;
				$tab_active = ( $status_filter === $tab_key ) ? 'mrme-status-tabs__item--active' : '';
			?>
			<li class="mrme-status-tabs__item <?php echo esc_attr( $tab_active ); ?>">
				<a href="<?php echo esc_url( $tab_url ); ?>"><?php echo esc_html( $tab_label ); ?></a>
			</li>
			<?php endforeach; ?>
		</ul>

		<form method="GET" class="mrme-search-form" action="<?php echo esc_url( $base_url ); ?>">
			<input type="hidden" name="page" value="mrme-campaigns">
			<?php if ( $status_filter ) : ?>
			<input type="hidden" name="status" value="<?php echo esc_attr( $status_filter ); ?>">
			<?php endif; ?>
			<input
				type="search"
				name="s"
				class="mrme-search-form__input"
				value="<?php echo esc_attr( $search ); ?>"
				placeholder="<?php esc_attr_e( 'Search campaigns…', 'myrock-mail-engine' ); ?>"
			>
			<button type="submit" class="button"><?php esc_html_e( 'Search', 'myrock-mail-engine' ); ?></button>
			<?php if ( $search ) : ?>
			<a href="<?php echo esc_url( $base_url ); ?>" class="button"><?php esc_html_e( 'Clear', 'myrock-mail-engine' ); ?></a>
			<?php endif; ?>
		</form>

	</div><!-- /.mrme-filter-bar -->

	<!-- Results summary -->
	<p class="mrme-results-summary">
		<?php
		printf(
			esc_html__( 'Showing %1$s of %2$s campaigns', 'myrock-mail-engine' ),
			'<strong>' . esc_html( number_format_i18n( count( $campaigns ) ) ) . '</strong>',
			'<strong>' . esc_html( number_format_i18n( $total ) ) . '</strong>'
		);
		?>
	</p>

	<?php if ( ! empty( $campaigns ) ) : ?>

	<form method="POST" id="mrme-campaigns-form" action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>">
		<?php wp_nonce_field( 'mrme_bulk_campaigns', '_wpnonce' ); ?>
		<input type="hidden" name="action" value="mrme_bulk_campaigns">

		<div class="mrme-bulk-action-bar">
			<select name="bulk_action" class="mrme-bulk-select">
				<option value=""><?php esc_html_e( 'Bulk Actions', 'myrock-mail-engine' ); ?></option>
				<option value="delete"><?php esc_html_e( 'Delete', 'myrock-mail-engine' ); ?></option>
				<option value="duplicate"><?php esc_html_e( 'Duplicate', 'myrock-mail-engine' ); ?></option>
			</select>
			<button type="submit" class="button"><?php esc_html_e( 'Apply', 'myrock-mail-engine' ); ?></button>
		</div>

		<table class="mrme-table widefat">
			<thead>
				<tr>
					<th class="mrme-table__check">
						<input type="checkbox" id="mrme-check-all" title="<?php esc_attr_e( 'Select all', 'myrock-mail-engine' ); ?>">
					</th>
					<th><?php esc_html_e( 'Title', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Status', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'From', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Scheduled / Sent', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Sent', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Opens', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Clicks', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Actions', 'myrock-mail-engine' ); ?></th>
				</tr>
			</thead>
			<tbody>
				<?php foreach ( $campaigns as $campaign ) :
					$c_id        = is_array( $campaign ) ? $campaign['id']         : $campaign->id;
					$c_title     = is_array( $campaign ) ? $campaign['title']       : $campaign->title;
					$c_status    = is_array( $campaign ) ? $campaign['status']      : $campaign->status;
					$c_from_name = is_array( $campaign ) ? ( $campaign['from_name']  ?? '' ) : ( $campaign->from_name  ?? '' );
					$c_from_email= is_array( $campaign ) ? ( $campaign['from_email'] ?? '' ) : ( $campaign->from_email ?? '' );
					$c_sent_at   = is_array( $campaign ) ? ( $campaign['sent_at']      ?? '' ) : ( $campaign->sent_at      ?? '' );
					$c_sched_at  = is_array( $campaign ) ? ( $campaign['scheduled_at'] ?? '' ) : ( $campaign->scheduled_at ?? '' );
					$c_stats_sent   = is_array( $campaign ) ? ( $campaign['stats']['sent']   ?? 0 ) : ( $campaign->stats['sent']   ?? 0 );
					$c_stats_opens  = is_array( $campaign ) ? ( $campaign['stats']['opens']  ?? 0 ) : ( $campaign->stats['opens']  ?? 0 );
					$c_stats_clicks = is_array( $campaign ) ? ( $campaign['stats']['clicks'] ?? 0 ) : ( $campaign->stats['clicks'] ?? 0 );

					$date_display = $c_sent_at ?: $c_sched_at;
					$edit_url      = add_query_arg( [ 'action' => 'edit',      'id' => (int) $c_id ], $base_url );
					$duplicate_url = add_query_arg( [ 'action' => 'duplicate', 'id' => (int) $c_id, '_wpnonce' => wp_create_nonce( 'mrme_duplicate_campaign_' . (int) $c_id ) ], $base_url );
					$send_url      = add_query_arg( [ 'action' => 'send',      'id' => (int) $c_id, '_wpnonce' => wp_create_nonce( 'mrme_send_campaign_' . (int) $c_id ) ], $base_url );
					$delete_url    = add_query_arg( [ 'action' => 'delete',    'id' => (int) $c_id, '_wpnonce' => wp_create_nonce( 'mrme_delete_campaign_' . (int) $c_id ) ], $base_url );
				?>
				<tr>
					<td class="mrme-table__check">
						<input type="checkbox" name="campaign_ids[]" value="<?php echo esc_attr( (int) $c_id ); ?>">
					</td>
					<td>
						<strong>
							<a href="<?php echo esc_url( $edit_url ); ?>"><?php echo esc_html( $c_title ); ?></a>
						</strong>
						<div class="mrme-row-actions">
							<a href="<?php echo esc_url( $edit_url ); ?>"><?php esc_html_e( 'Edit', 'myrock-mail-engine' ); ?></a>
							|
							<a href="<?php echo esc_url( $duplicate_url ); ?>"><?php esc_html_e( 'Duplicate', 'myrock-mail-engine' ); ?></a>
							<?php if ( in_array( $c_status, [ 'draft', 'scheduled' ], true ) ) : ?>
							|
							<a
								href="<?php echo esc_url( $delete_url ); ?>"
								class="mrme-action--delete"
								data-confirm="<?php esc_attr_e( 'Delete this campaign? This action cannot be undone.', 'myrock-mail-engine' ); ?>"
							><?php esc_html_e( 'Delete', 'myrock-mail-engine' ); ?></a>
							<?php endif; ?>
						</div>
					</td>
					<td>
						<span class="mrme-badge mrme-badge--<?php echo esc_attr( $c_status ); ?>">
							<?php echo esc_html( ucfirst( $c_status ) ); ?>
						</span>
					</td>
					<td>
						<?php if ( $c_from_name || $c_from_email ) : ?>
						<span><?php echo esc_html( $c_from_name ); ?></span>
						<br>
						<small class="mrme-muted"><?php echo esc_html( $c_from_email ); ?></small>
						<?php else : ?>
						<span class="mrme-muted">—</span>
						<?php endif; ?>
					</td>
					<td>
						<?php echo esc_html( $date_display
							? wp_date( get_option( 'date_format' ) . ' H:i', strtotime( $date_display ) )
							: '—'
						); ?>
					</td>
					<td><?php echo esc_html( number_format_i18n( (int) $c_stats_sent ) ); ?></td>
					<td><?php echo esc_html( number_format_i18n( (int) $c_stats_opens ) ); ?></td>
					<td><?php echo esc_html( number_format_i18n( (int) $c_stats_clicks ) ); ?></td>
					<td class="mrme-table__actions">
						<a href="<?php echo esc_url( $edit_url ); ?>" class="button button-small"><?php esc_html_e( 'Edit', 'myrock-mail-engine' ); ?></a>
						<?php if ( 'draft' === $c_status || 'scheduled' === $c_status ) : ?>
						<a
							href="<?php echo esc_url( $send_url ); ?>"
							class="button button-small button-primary mrme-action--send"
							data-confirm="<?php esc_attr_e( 'Send this campaign now to all recipients?', 'myrock-mail-engine' ); ?>"
						><?php esc_html_e( 'Send', 'myrock-mail-engine' ); ?></a>
						<?php endif; ?>
					</td>
				</tr>
				<?php endforeach; ?>
			</tbody>
		</table>

	</form>

	<?php else : ?>
	<div class="mrme-empty-state">
		<span class="dashicons dashicons-email-alt mrme-empty-state__icon"></span>
		<p>
			<?php $search
				? esc_html_e( 'No campaigns match your search.', 'myrock-mail-engine' )
				: esc_html_e( 'No campaigns yet. Create your first campaign.', 'myrock-mail-engine' );
			?>
		</p>
		<a href="<?php echo esc_url( add_query_arg( 'action', 'new', $base_url ) ); ?>" class="button button-primary">
			<?php esc_html_e( '+ Create Campaign', 'myrock-mail-engine' ); ?>
		</a>
	</div>
	<?php endif; ?>

	<!-- Pagination -->
	<?php if ( $total_pages > 1 ) : ?>
	<div class="mrme-pagination">
		<?php
		$paginate_args = [
			'base'      => add_query_arg( 'paged', '%#%', $base_url ),
			'format'    => '',
			'current'   => $paged,
			'total'     => $total_pages,
			'prev_text' => '&laquo; ' . esc_html__( 'Previous', 'myrock-mail-engine' ),
			'next_text' => esc_html__( 'Next', 'myrock-mail-engine' ) . ' &raquo;',
		];
		if ( $status_filter ) {
			$paginate_args['base'] = add_query_arg( 'status', $status_filter, $paginate_args['base'] );
		}
		if ( $search ) {
			$paginate_args['base'] = add_query_arg( 's', $search, $paginate_args['base'] );
		}
		echo wp_kses_post( paginate_links( $paginate_args ) );
		?>
	</div>
	<?php endif; ?>

</div><!-- /.wrap.mrme-wrap -->
