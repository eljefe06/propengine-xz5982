<?php
/**
 * Admin template: Contacts list.
 *
 * Available variables (set by ContactsPage before including this template):
 *   @var array  $contacts       Paged array of contact rows.
 *   @var int    $total          Total contact count for current filter.
 *   @var int    $paged          Current page number.
 *   @var int    $per_page       Rows per page.
 *   @var array  $lists          All available lists (for filter / import modal).
 *   @var array  $tags           All available tags.
 *   @var string $status_filter  Active status tab ('', 'subscribed', …).
 *   @var string $search         Active search string.
 *
 * @package MyRock\MailEngine
 */

defined( 'ABSPATH' ) || exit;

$total_pages    = $per_page > 0 ? (int) ceil( $total / $per_page ) : 1;
$base_url       = admin_url( 'admin.php?page=mrme-contacts' );
$status_tabs    = [
	''             => __( 'All', 'myrock-mail-engine' ),
	'subscribed'   => __( 'Subscribed', 'myrock-mail-engine' ),
	'unsubscribed' => __( 'Unsubscribed', 'myrock-mail-engine' ),
	'pending'      => __( 'Pending', 'myrock-mail-engine' ),
	'bounced'      => __( 'Bounced', 'myrock-mail-engine' ),
];
?>
<div class="wrap mrme-wrap">

	<!-- Page header -->
	<div class="mrme-page-header">
		<h1 class="mrme-page-title"><?php esc_html_e( 'Contacts', 'myrock-mail-engine' ); ?></h1>
		<div class="mrme-page-header__actions">
			<a href="<?php echo esc_url( add_query_arg( 'action', 'new', $base_url ) ); ?>" class="button button-primary">
				<?php esc_html_e( '+ Add Contact', 'myrock-mail-engine' ); ?>
			</a>
			<button type="button" class="button" id="mrme-open-import-modal">
				<?php esc_html_e( 'Import CSV', 'myrock-mail-engine' ); ?>
			</button>
		</div>
	</div>

	<!-- Filter bar -->
	<div class="mrme-filter-bar">

		<!-- Status tabs -->
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

		<!-- Search form -->
		<form method="GET" class="mrme-search-form" action="<?php echo esc_url( $base_url ); ?>">
			<input type="hidden" name="page" value="mrme-contacts">
			<?php if ( $status_filter ) : ?>
			<input type="hidden" name="status" value="<?php echo esc_attr( $status_filter ); ?>">
			<?php endif; ?>
			<input
				type="search"
				name="s"
				class="mrme-search-form__input"
				value="<?php echo esc_attr( $search ); ?>"
				placeholder="<?php esc_attr_e( 'Search by name or email…', 'myrock-mail-engine' ); ?>"
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
			/* translators: 1: number of contacts shown, 2: total */
			esc_html__( 'Showing %1$s of %2$s contacts', 'myrock-mail-engine' ),
			'<strong>' . esc_html( number_format_i18n( count( $contacts ) ) ) . '</strong>',
			'<strong>' . esc_html( number_format_i18n( $total ) ) . '</strong>'
		);
		?>
	</p>

	<!-- Bulk action form -->
	<form method="POST" id="mrme-contacts-form" action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>">
		<?php wp_nonce_field( 'mrme_bulk_contacts', '_wpnonce' ); ?>
		<input type="hidden" name="action" value="mrme_bulk_contacts">

		<div class="mrme-bulk-action-bar">
			<select name="bulk_action" class="mrme-bulk-select">
				<option value=""><?php esc_html_e( 'Bulk Actions', 'myrock-mail-engine' ); ?></option>
				<option value="delete"><?php esc_html_e( 'Delete', 'myrock-mail-engine' ); ?></option>
				<option value="unsubscribe"><?php esc_html_e( 'Unsubscribe', 'myrock-mail-engine' ); ?></option>
				<option value="subscribe"><?php esc_html_e( 'Mark Subscribed', 'myrock-mail-engine' ); ?></option>
			</select>
			<button type="submit" class="button"><?php esc_html_e( 'Apply', 'myrock-mail-engine' ); ?></button>
		</div>

		<?php if ( ! empty( $contacts ) ) : ?>
		<table class="mrme-table widefat">
			<thead>
				<tr>
					<th class="mrme-table__check">
						<input type="checkbox" id="mrme-check-all" title="<?php esc_attr_e( 'Select all', 'myrock-mail-engine' ); ?>">
					</th>
					<th><?php esc_html_e( 'Name', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Email', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Status', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Lists', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Tags', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Date Added', 'myrock-mail-engine' ); ?></th>
					<th><?php esc_html_e( 'Actions', 'myrock-mail-engine' ); ?></th>
				</tr>
			</thead>
			<tbody>
				<?php foreach ( $contacts as $contact ) :
					$c_id         = is_array( $contact ) ? $contact['id']         : $contact->id;
					$c_first      = is_array( $contact ) ? ( $contact['first_name'] ?? '' ) : ( $contact->first_name ?? '' );
					$c_last       = is_array( $contact ) ? ( $contact['last_name']  ?? '' ) : ( $contact->last_name  ?? '' );
					$c_email      = is_array( $contact ) ? $contact['email']       : $contact->email;
					$c_status     = is_array( $contact ) ? $contact['status']      : $contact->status;
					$c_lists      = is_array( $contact ) ? ( $contact['list_names'] ?? [] ) : ( $contact->list_names ?? [] );
					$c_tags       = is_array( $contact ) ? ( $contact['tag_names']  ?? [] ) : ( $contact->tag_names  ?? [] );
					$c_created    = is_array( $contact ) ? ( $contact['created_at'] ?? '' ) : ( $contact->created_at ?? '' );
					$c_name       = trim( $c_first . ' ' . $c_last ) ?: $c_email;
					$edit_url     = add_query_arg( [ 'action' => 'edit', 'id' => (int) $c_id ], $base_url );
					$delete_url   = add_query_arg( [ 'action' => 'delete', 'id' => (int) $c_id, '_wpnonce' => wp_create_nonce( 'mrme_delete_contact_' . (int) $c_id ) ], $base_url );
				?>
				<tr>
					<td class="mrme-table__check">
						<input type="checkbox" name="contact_ids[]" value="<?php echo esc_attr( (int) $c_id ); ?>">
					</td>
					<td>
						<strong>
							<a href="<?php echo esc_url( $edit_url ); ?>"><?php echo esc_html( $c_name ); ?></a>
						</strong>
					</td>
					<td><?php echo esc_html( $c_email ); ?></td>
					<td>
						<span class="mrme-badge mrme-badge--<?php echo esc_attr( $c_status ); ?>">
							<?php echo esc_html( ucfirst( $c_status ) ); ?>
						</span>
					</td>
					<td>
						<?php if ( ! empty( $c_lists ) ) :
							$display_lists = is_array( $c_lists ) ? $c_lists : [ $c_lists ];
							echo esc_html( implode( ', ', array_map( 'strval', $display_lists ) ) );
						else : ?>
						<span class="mrme-muted">—</span>
						<?php endif; ?>
					</td>
					<td>
						<?php if ( ! empty( $c_tags ) ) :
							$display_tags = is_array( $c_tags ) ? $c_tags : [ $c_tags ];
							foreach ( $display_tags as $tag_name ) : ?>
							<span class="mrme-tag"><?php echo esc_html( $tag_name ); ?></span>
							<?php endforeach;
						else : ?>
						<span class="mrme-muted">—</span>
						<?php endif; ?>
					</td>
					<td>
						<?php echo esc_html(
							$c_created
								? wp_date( get_option( 'date_format' ), strtotime( $c_created ) )
								: '—'
						); ?>
					</td>
					<td class="mrme-table__actions">
						<a href="<?php echo esc_url( $edit_url ); ?>"><?php esc_html_e( 'Edit', 'myrock-mail-engine' ); ?></a>
						|
						<a
							href="<?php echo esc_url( $delete_url ); ?>"
							class="mrme-action--delete"
							data-confirm="<?php esc_attr_e( 'Delete this contact? This action cannot be undone.', 'myrock-mail-engine' ); ?>"
						><?php esc_html_e( 'Delete', 'myrock-mail-engine' ); ?></a>
					</td>
				</tr>
				<?php endforeach; ?>
			</tbody>
		</table>

		<?php else : ?>
		<div class="mrme-empty-state">
			<span class="dashicons dashicons-groups mrme-empty-state__icon"></span>
			<p>
				<?php
				$search
					? esc_html_e( 'No contacts match your search.', 'myrock-mail-engine' )
					: esc_html_e( 'No contacts yet. Add your first contact or import a CSV file.', 'myrock-mail-engine' );
				?>
			</p>
		</div>
		<?php endif; ?>

	</form><!-- /#mrme-contacts-form -->

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
		// Additional params.
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

<!-- ================================================================== -->
<!-- Import CSV Modal                                                     -->
<!-- ================================================================== -->
<div id="mrme-import-modal" class="mrme-modal" aria-hidden="true" role="dialog" aria-labelledby="mrme-import-modal-title">
	<div class="mrme-modal__overlay" data-modal-close></div>
	<div class="mrme-modal__content">

		<div class="mrme-modal__header">
			<h2 id="mrme-import-modal-title"><?php esc_html_e( 'Import Contacts from CSV', 'myrock-mail-engine' ); ?></h2>
			<button type="button" class="mrme-modal__close" data-modal-close aria-label="<?php esc_attr_e( 'Close', 'myrock-mail-engine' ); ?>">&times;</button>
		</div>

		<form
			method="POST"
			enctype="multipart/form-data"
			action="<?php echo esc_url( admin_url( 'admin-post.php' ) ); ?>"
			class="mrme-import-form"
		>
			<?php wp_nonce_field( 'mrme_import_contacts', '_wpnonce' ); ?>
			<input type="hidden" name="action" value="mrme_import_contacts">

			<div class="mrme-form-section">
				<label for="mrme-import-file" class="mrme-form-section__label">
					<?php esc_html_e( 'CSV File', 'myrock-mail-engine' ); ?> <span aria-hidden="true">*</span>
				</label>
				<p class="description">
					<?php esc_html_e( 'File must have a header row with columns: email, first_name, last_name, phone, company.', 'myrock-mail-engine' ); ?>
				</p>
				<input
					type="file"
					id="mrme-import-file"
					name="import_csv"
					accept=".csv,text/csv"
					required
					class="mrme-form-section__file"
				>
			</div>

			<div class="mrme-form-section">
				<fieldset>
					<legend class="mrme-form-section__label"><?php esc_html_e( 'Assign to Lists', 'myrock-mail-engine' ); ?></legend>
					<div class="mrme-checkbox-grid">
						<?php if ( ! empty( $lists ) ) :
							foreach ( $lists as $list ) :
								$l_id   = is_array( $list ) ? $list['id']   : $list->id;
								$l_name = is_array( $list ) ? $list['name'] : $list->name;
						?>
						<label class="mrme-checkbox-grid__item">
							<input type="checkbox" name="import_list_ids[]" value="<?php echo esc_attr( (int) $l_id ); ?>">
							<?php echo esc_html( $l_name ); ?>
						</label>
						<?php endforeach;
						else : ?>
						<p class="mrme-muted"><?php esc_html_e( 'No lists found. Create a list first.', 'myrock-mail-engine' ); ?></p>
						<?php endif; ?>
					</div>
				</fieldset>
			</div>

			<div class="mrme-form-section">
				<label class="mrme-checkbox-inline">
					<input type="checkbox" name="update_existing" value="1">
					<?php esc_html_e( 'Update existing contacts if email already exists', 'myrock-mail-engine' ); ?>
				</label>
			</div>

			<div class="mrme-modal__footer">
				<button type="submit" class="button button-primary">
					<?php esc_html_e( 'Import', 'myrock-mail-engine' ); ?>
				</button>
				<button type="button" class="button" data-modal-close>
					<?php esc_html_e( 'Cancel', 'myrock-mail-engine' ); ?>
				</button>
			</div>

		</form>

	</div><!-- /.mrme-modal__content -->
</div><!-- /#mrme-import-modal -->
