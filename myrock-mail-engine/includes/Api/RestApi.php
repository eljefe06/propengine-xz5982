<?php
namespace MyRock\MailEngine\Api;
/**
 * RestApi — registers and handles all REST API endpoints for MyRock Mail Engine.
 *
 * @package MyRock\MailEngine\Api
 */

defined( 'ABSPATH' ) || exit;


use WP_REST_Request;
use WP_REST_Response;
use WP_Error;
use MyRock\MailEngine\Models\Contact;
use MyRock\MailEngine\Models\MailList;
use MyRock\MailEngine\Models\Campaign;
use MyRock\MailEngine\Models\Form;
use MyRock\MailEngine\Services\ContactService;

/**
 * Class RestApi
 *
 * All routes are grouped under the 'mrme/v1' namespace.
 * Authentication accepts either:
 *   - A logged-in user with the manage_options capability, OR
 *   - A valid API key sent in the X-MRME-Key HTTP header.
 */
class RestApi {

	/** REST namespace. */
	const NAMESPACE = 'mrme/v1';

	/**
	 * Register all REST routes.
	 *
	 * @return void
	 */
	public function register_routes(): void {

		$perm = [ $this, 'permission_callback' ];

		// ------------------------------------------------------------------ //
		// Contacts.
		// ------------------------------------------------------------------ //
		register_rest_route( self::NAMESPACE, '/contacts', [
			[
				'methods'             => \WP_REST_Server::READABLE,
				'callback'            => [ $this, 'get_contacts' ],
				'permission_callback' => $perm,
				'args'                => $this->contacts_collection_args(),
			],
			[
				'methods'             => \WP_REST_Server::CREATABLE,
				'callback'            => [ $this, 'create_contact' ],
				'permission_callback' => $perm,
				'args'                => $this->contact_create_args(),
			],
		] );

		register_rest_route( self::NAMESPACE, '/contacts/(?P<id>\d+)', [
			[
				'methods'             => \WP_REST_Server::READABLE,
				'callback'            => [ $this, 'get_contact' ],
				'permission_callback' => $perm,
				'args'                => [ 'id' => [ 'validate_callback' => 'is_numeric' ] ],
			],
			[
				'methods'             => \WP_REST_Server::EDITABLE,
				'callback'            => [ $this, 'update_contact' ],
				'permission_callback' => $perm,
				'args'                => $this->contact_update_args(),
			],
			[
				'methods'             => \WP_REST_Server::DELETABLE,
				'callback'            => [ $this, 'delete_contact' ],
				'permission_callback' => $perm,
				'args'                => [ 'id' => [ 'validate_callback' => 'is_numeric' ] ],
			],
		] );

		// ------------------------------------------------------------------ //
		// Lists.
		// ------------------------------------------------------------------ //
		register_rest_route( self::NAMESPACE, '/lists', [
			[
				'methods'             => \WP_REST_Server::READABLE,
				'callback'            => [ $this, 'get_lists' ],
				'permission_callback' => $perm,
			],
		] );

		// ------------------------------------------------------------------ //
		// Campaigns.
		// ------------------------------------------------------------------ //
		register_rest_route( self::NAMESPACE, '/campaigns', [
			[
				'methods'             => \WP_REST_Server::READABLE,
				'callback'            => [ $this, 'get_campaigns' ],
				'permission_callback' => $perm,
			],
		] );

		// ------------------------------------------------------------------ //
		// Forms.
		// ------------------------------------------------------------------ //
		register_rest_route( self::NAMESPACE, '/forms', [
			[
				'methods'             => \WP_REST_Server::READABLE,
				'callback'            => [ $this, 'get_forms' ],
				'permission_callback' => $perm,
			],
		] );
	}

	// ------------------------------------------------------------------ //
	// Permission callback.
	// ------------------------------------------------------------------ //

	/**
	 * Check whether the incoming request is authorised.
	 *
	 * Accepts:
	 *  - current_user_can( 'manage_options' ), OR
	 *  - Header X-MRME-Key matching the mrme_api_key option.
	 *
	 * @param WP_REST_Request $r Incoming REST request.
	 * @return bool
	 */
	public function permission_callback( WP_REST_Request $r ): bool {

		// Logged-in admin.
		if ( current_user_can( 'manage_options' ) ) {
			return true;
		}

		// API-key authentication.
		$stored_key = get_option( 'mrme_api_key', '' );

		if ( ! empty( $stored_key ) ) {
			$request_key = $r->get_header( 'X-MRME-Key' );
			if ( ! empty( $request_key ) && hash_equals( $stored_key, $request_key ) ) {
				return true;
			}
		}

		return false;
	}

	// ------------------------------------------------------------------ //
	// Contacts: collection.
	// ------------------------------------------------------------------ //

	/**
	 * GET /contacts — paginated list of contacts.
	 *
	 * Supported query params: search, status, list_id, per_page (max 100), page.
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return WP_REST_Response
	 */
	public function get_contacts( WP_REST_Request $r ): WP_REST_Response {

		$per_page = min( (int) $r->get_param( 'per_page' ) ?: 25, 100 );
		$page     = max( (int) $r->get_param( 'page' ) ?: 1, 1 );
		$search   = sanitize_text_field( $r->get_param( 'search' ) ?: '' );
		$status   = sanitize_key( $r->get_param( 'status' ) ?: '' );
		$list_id  = (int) ( $r->get_param( 'list_id' ) ?: 0 );

		$filter_args = [
			'search'  => $search,
			'status'  => $status,
			'list_id' => $list_id,
		];

		$contacts = Contact::all( array_merge( $filter_args, [
			'limit'  => $per_page,
			'offset' => ( $page - 1 ) * $per_page,
		] ) );

		$total = Contact::count( $filter_args );
		$pages = $per_page > 0 ? (int) ceil( $total / $per_page ) : 1;

		return new WP_REST_Response(
			[
				'data'  => array_map( [ $this, 'format_contact' ], $contacts ),
				'total' => $total,
				'pages' => $pages,
			],
			200
		);
	}

	/**
	 * POST /contacts — create a new contact.
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return WP_REST_Response|WP_Error
	 */
	public function create_contact( WP_REST_Request $r ) {

		$data = $this->extract_contact_fields( $r );

		if ( empty( $data['email'] ) || ! is_email( $data['email'] ) ) {
			return new WP_Error(
				'mrme_invalid_email',
				__( 'A valid email address is required.', 'myrock-mail-engine' ),
				[ 'status' => 422 ]
			);
		}

		$contact_id = ContactService::create( $data );

		if ( ! $contact_id ) {
			return new WP_Error(
				'mrme_create_failed',
				__( 'Could not create contact.', 'myrock-mail-engine' ),
				[ 'status' => 500 ]
			);
		}

		$contact = Contact::find( $contact_id );

		return new WP_REST_Response( $this->format_contact( $contact ), 201 );
	}

	// ------------------------------------------------------------------ //
	// Contacts: single item.
	// ------------------------------------------------------------------ //

	/**
	 * GET /contacts/{id} — retrieve a single contact.
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return WP_REST_Response|WP_Error
	 */
	public function get_contact( WP_REST_Request $r ) {

		$contact = Contact::find( (int) $r->get_param( 'id' ) );

		if ( ! $contact ) {
			return new WP_Error(
				'mrme_not_found',
				__( 'Contact not found.', 'myrock-mail-engine' ),
				[ 'status' => 404 ]
			);
		}

		return new WP_REST_Response( $this->format_contact( $contact ), 200 );
	}

	/**
	 * PUT /contacts/{id} — update an existing contact.
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return WP_REST_Response|WP_Error
	 */
	public function update_contact( WP_REST_Request $r ) {

		$id = (int) $r->get_param( 'id' );

		$existing = Contact::find( $id );

		if ( ! $existing ) {
			return new WP_Error(
				'mrme_not_found',
				__( 'Contact not found.', 'myrock-mail-engine' ),
				[ 'status' => 404 ]
			);
		}

		$data       = $this->extract_contact_fields( $r );
		$data['id'] = $id;

		// If no email provided, keep the existing one.
		if ( empty( $data['email'] ) ) {
			$data['email'] = is_array( $existing ) ? $existing['email'] : $existing->email;
		}

		$updated = ContactService::update( $id, $data );

		if ( ! $updated ) {
			return new WP_Error(
				'mrme_update_failed',
				__( 'Could not update contact.', 'myrock-mail-engine' ),
				[ 'status' => 500 ]
			);
		}

		$contact = Contact::find( $id );

		return new WP_REST_Response( $this->format_contact( $contact ), 200 );
	}

	/**
	 * DELETE /contacts/{id} — permanently delete a contact.
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return WP_REST_Response|WP_Error
	 */
	public function delete_contact( WP_REST_Request $r ) {

		$id = (int) $r->get_param( 'id' );

		$existing = Contact::find( $id );

		if ( ! $existing ) {
			return new WP_Error(
				'mrme_not_found',
				__( 'Contact not found.', 'myrock-mail-engine' ),
				[ 'status' => 404 ]
			);
		}

		$deleted = Contact::delete( $id );

		if ( ! $deleted ) {
			return new WP_Error(
				'mrme_delete_failed',
				__( 'Could not delete contact.', 'myrock-mail-engine' ),
				[ 'status' => 500 ]
			);
		}

		return new WP_REST_Response( [ 'deleted' => true, 'id' => $id ], 200 );
	}

	// ------------------------------------------------------------------ //
	// Lists, Campaigns, Forms.
	// ------------------------------------------------------------------ //

	/**
	 * GET /lists — return all mailing lists.
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return WP_REST_Response
	 */
	public function get_lists( WP_REST_Request $r ): WP_REST_Response {

		$lists = MailList::all();

		if ( ! is_array( $lists ) ) {
			$lists = [];
		}

		return new WP_REST_Response( [ 'data' => $lists ], 200 );
	}

	/**
	 * GET /campaigns — return all campaigns (supports per_page + page).
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return WP_REST_Response
	 */
	public function get_campaigns( WP_REST_Request $r ): WP_REST_Response {

		$per_page = min( (int) $r->get_param( 'per_page' ) ?: 25, 100 );
		$page     = max( (int) $r->get_param( 'page' ) ?: 1, 1 );
		$status   = sanitize_key( $r->get_param( 'status' ) ?: '' );

		$args = [
			'per_page' => $per_page,
			'page'     => $page,
			'status'   => $status,
		];

		$campaigns = Campaign::all( [
			'status' => $status,
			'limit'  => $per_page,
			'offset' => ( $page - 1 ) * $per_page,
		] );
		$total     = Campaign::count( [ 'status' => $status ] );
		$pages     = $per_page > 0 ? (int) ceil( $total / $per_page ) : 1;

		return new WP_REST_Response(
			[
				'data'  => $campaigns,
				'total' => $total,
				'pages' => $pages,
			],
			200
		);
	}

	/**
	 * GET /forms — return all subscription forms.
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return WP_REST_Response
	 */
	public function get_forms( WP_REST_Request $r ): WP_REST_Response {

		$forms = Form::all();

		if ( ! is_array( $forms ) ) {
			$forms = [];
		}

		return new WP_REST_Response( [ 'data' => $forms ], 200 );
	}

	// ------------------------------------------------------------------ //
	// Private helpers.
	// ------------------------------------------------------------------ //

	/**
	 * Format a contact record for API output.
	 *
	 * Accepts both array and object shapes returned by the Contact model.
	 *
	 * @param array|object|null $contact Raw contact data.
	 * @return array Formatted contact array.
	 */
	private function format_contact( $contact ): array {

		if ( ! $contact ) {
			return [];
		}

		$get = static function ( $contact, string $key, $default = null ) {
			if ( is_array( $contact ) ) {
				return $contact[ $key ] ?? $default;
			}
			return $contact->$key ?? $default;
		};

		return [
			'id'         => (int) $get( $contact, 'id' ),
			'email'      => $get( $contact, 'email', '' ),
			'first_name' => $get( $contact, 'first_name', '' ),
			'last_name'  => $get( $contact, 'last_name', '' ),
			'phone'      => $get( $contact, 'phone', '' ),
			'company'    => $get( $contact, 'company', '' ),
			'status'     => $get( $contact, 'status', 'subscribed' ),
			'source'     => $get( $contact, 'source', '' ),
			'ip_address' => $get( $contact, 'ip_address', '' ),
			'notes'      => $get( $contact, 'notes', '' ),
			'meta'       => $get( $contact, 'meta', [] ),
			'list_ids'   => $get( $contact, 'list_ids', [] ),
			'tag_ids'    => $get( $contact, 'tag_ids', [] ),
			'created_at' => $get( $contact, 'created_at', '' ),
			'updated_at' => $get( $contact, 'updated_at', '' ),
		];
	}

	/**
	 * Extract and sanitise contact fields from a REST request body.
	 *
	 * @param WP_REST_Request $r Request object.
	 * @return array Sanitised field array.
	 */
	private function extract_contact_fields( WP_REST_Request $r ): array {

		$email      = sanitize_email( $r->get_param( 'email' ) ?: '' );
		$first_name = sanitize_text_field( $r->get_param( 'first_name' ) ?: '' );
		$last_name  = sanitize_text_field( $r->get_param( 'last_name' ) ?: '' );
		$phone      = sanitize_text_field( $r->get_param( 'phone' ) ?: '' );
		$company    = sanitize_text_field( $r->get_param( 'company' ) ?: '' );
		$status     = sanitize_key( $r->get_param( 'status' ) ?: 'subscribed' );
		$source     = sanitize_text_field( $r->get_param( 'source' ) ?: 'api' );
		$notes      = sanitize_textarea_field( $r->get_param( 'notes' ) ?: '' );

		$list_ids = $r->get_param( 'list_ids' );
		$list_ids = is_array( $list_ids ) ? array_map( 'intval', $list_ids ) : [];

		$tag_ids = $r->get_param( 'tag_ids' );
		$tag_ids = is_array( $tag_ids ) ? array_map( 'intval', $tag_ids ) : [];

		$meta = $r->get_param( 'meta' );
		$meta = is_array( $meta ) ? array_map( 'sanitize_text_field', $meta ) : [];

		return [
			'email'      => $email,
			'first_name' => $first_name,
			'last_name'  => $last_name,
			'phone'      => $phone,
			'company'    => $company,
			'status'     => $status,
			'source'     => $source,
			'notes'      => $notes,
			'list_ids'   => $list_ids,
			'tag_ids'    => $tag_ids,
			'meta'       => $meta,
		];
	}

	/**
	 * REST argument schema for the contact collection endpoint.
	 *
	 * @return array
	 */
	private function contacts_collection_args(): array {
		return [
			'search'   => [
				'type'              => 'string',
				'sanitize_callback' => 'sanitize_text_field',
				'default'           => '',
			],
			'status'   => [
				'type'              => 'string',
				'sanitize_callback' => 'sanitize_key',
				'default'           => '',
				'enum'              => [ '', 'subscribed', 'unsubscribed', 'pending', 'bounced' ],
			],
			'list_id'  => [
				'type'              => 'integer',
				'sanitize_callback' => 'absint',
				'default'           => 0,
			],
			'per_page' => [
				'type'              => 'integer',
				'sanitize_callback' => 'absint',
				'default'           => 25,
				'minimum'           => 1,
				'maximum'           => 100,
			],
			'page'     => [
				'type'              => 'integer',
				'sanitize_callback' => 'absint',
				'default'           => 1,
				'minimum'           => 1,
			],
		];
	}

	/**
	 * REST argument schema for the contact creation endpoint.
	 *
	 * @return array
	 */
	private function contact_create_args(): array {
		return [
			'email'      => [
				'type'              => 'string',
				'format'            => 'email',
				'required'          => true,
				'sanitize_callback' => 'sanitize_email',
			],
			'first_name' => [ 'type' => 'string', 'sanitize_callback' => 'sanitize_text_field' ],
			'last_name'  => [ 'type' => 'string', 'sanitize_callback' => 'sanitize_text_field' ],
			'phone'      => [ 'type' => 'string', 'sanitize_callback' => 'sanitize_text_field' ],
			'company'    => [ 'type' => 'string', 'sanitize_callback' => 'sanitize_text_field' ],
			'status'     => [
				'type'              => 'string',
				'sanitize_callback' => 'sanitize_key',
				'default'           => 'subscribed',
				'enum'              => [ 'subscribed', 'unsubscribed', 'pending', 'bounced' ],
			],
			'source'     => [ 'type' => 'string', 'sanitize_callback' => 'sanitize_text_field' ],
			'notes'      => [ 'type' => 'string', 'sanitize_callback' => 'sanitize_textarea_field' ],
			'list_ids'   => [ 'type' => 'array', 'items' => [ 'type' => 'integer' ] ],
			'tag_ids'    => [ 'type' => 'array', 'items' => [ 'type' => 'integer' ] ],
			'meta'       => [ 'type' => 'object' ],
		];
	}

	/**
	 * REST argument schema for the contact update endpoint.
	 *
	 * Same as create but email is not required.
	 *
	 * @return array
	 */
	private function contact_update_args(): array {
		$args          = $this->contact_create_args();
		$args['email']['required'] = false;
		return $args;
	}
}
