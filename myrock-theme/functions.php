<?php
/**
 * MyRock Theme — functions.php
 * Child theme of Twenty Twenty-Five.
 *
 * @package MyRock_Theme
 */

defined( 'ABSPATH' ) || exit;

/* ============================================================
   ENQUEUE STYLES + FONTS
   ============================================================ */
add_action( 'wp_enqueue_scripts', 'myrock_theme_enqueue' );
function myrock_theme_enqueue() {
	// Google Fonts — match the static site exactly
	wp_enqueue_style(
		'myrock-fonts',
		'https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500&family=DM+Mono&display=swap',
		[],
		null
	);

	// Child theme styles
	wp_enqueue_style(
		'myrock-theme-style',
		get_stylesheet_directory_uri() . '/assets/blog.css',
		[ 'myrock-fonts' ],
		'1.0.0'
	);
}

/* ============================================================
   THEME SUPPORT
   ============================================================ */
add_action( 'after_setup_theme', 'myrock_theme_setup' );
function myrock_theme_setup() {
	add_theme_support( 'post-thumbnails' );
	add_theme_support( 'automatic-feed-links' );
	add_theme_support( 'title-tag' );
	add_theme_support( 'custom-logo' );

	add_image_size( 'myrock-card',   800, 500, true );
	add_image_size( 'myrock-hero',  1600, 800, true );
	add_image_size( 'myrock-thumb',  400, 300, true );

	// Menus
	register_nav_menus( [
		'blog-nav'    => __( 'Blog Navigation', 'myrock-theme' ),
		'blog-footer' => __( 'Blog Footer',     'myrock-theme' ),
	] );
}

/* ============================================================
   READING TIME
   ============================================================ */
function myrock_reading_time( $post_id = null ) {
	$post_id  = $post_id ?: get_the_ID();
	$content  = get_post_field( 'post_content', $post_id );
	$content  = strip_shortcodes( wp_strip_all_tags( $content ) );
	$words    = str_word_count( $content );
	$minutes  = max( 1, (int) ceil( $words / 200 ) );
	return sprintf( _n( '%d min de lectura', '%d min de lectura', $minutes, 'myrock-theme' ), $minutes );
}

/* ============================================================
   BREADCRUMBS
   ============================================================ */
function myrock_breadcrumbs() {
	$home_url  = home_url( '/blog/' );
	$separator = '<span class="mr-bc-sep" aria-hidden="true">/</span>';
	$crumbs    = [];

	$crumbs[] = '<a href="' . esc_url( home_url( '/' ) ) . '" class="mr-bc-link">MyRock</a>';
	$crumbs[] = '<a href="' . esc_url( $home_url ) . '" class="mr-bc-link">Blog</a>';

	if ( is_single() ) {
		$categories = get_the_category();
		if ( $categories ) {
			$cat = $categories[0];
			$crumbs[] = '<a href="' . esc_url( get_category_link( $cat->term_id ) ) . '" class="mr-bc-link">'
				. esc_html( $cat->name ) . '</a>';
		}
		$crumbs[] = '<span class="mr-bc-current">' . get_the_title() . '</span>';
	} elseif ( is_category() ) {
		$crumbs[] = '<span class="mr-bc-current">' . single_cat_title( '', false ) . '</span>';
	} elseif ( is_tag() ) {
		$crumbs[] = '<span class="mr-bc-current">' . single_tag_title( '', false ) . '</span>';
	} elseif ( is_search() ) {
		$crumbs[] = '<span class="mr-bc-current">Búsqueda: ' . get_search_query() . '</span>';
	}

	echo '<nav class="mr-breadcrumbs" aria-label="Breadcrumb">'
		. implode( ' ' . $separator . ' ', $crumbs )
		. '</nav>';
}

/* ============================================================
   RELATED POSTS
   ============================================================ */
function myrock_related_posts( $post_id = null, $count = 3 ) {
	$post_id    = $post_id ?: get_the_ID();
	$categories = wp_get_post_categories( $post_id );

	if ( empty( $categories ) ) return [];

	$args = [
		'category__in'       => $categories,
		'post__not_in'       => [ $post_id ],
		'posts_per_page'     => $count,
		'orderby'            => 'rand',
		'post_status'        => 'publish',
		'ignore_sticky_posts'=> 1,
	];

	return get_posts( $args );
}

/* ============================================================
   EXCERPT LENGTH
   ============================================================ */
add_filter( 'excerpt_length', fn() => 22 );
add_filter( 'excerpt_more',   fn() => '…' );

/* ============================================================
   CUSTOM BODY CLASSES
   ============================================================ */
add_filter( 'body_class', function( $classes ) {
	$classes[] = 'myrock-blog';
	return $classes;
} );

/* ============================================================
   DISABLE BLOCK EDITOR GLOBAL STYLES (avoid TT25 color override)
   ============================================================ */
add_action( 'wp_enqueue_scripts', function() {
	wp_dequeue_style( 'global-styles' );
}, 100 );

/* ============================================================
   SEO META TAGS (basic, no plugin required)
   ============================================================ */
add_action( 'wp_head', 'myrock_seo_meta', 1 );
function myrock_seo_meta() {
	if ( is_singular() ) {
		$post        = get_queried_object();
		$description = wp_strip_all_tags( get_the_excerpt( $post->ID ) );
		$thumbnail   = get_the_post_thumbnail_url( $post->ID, 'myrock-hero' );

		echo '<meta name="description" content="' . esc_attr( $description ) . '">' . "\n";
		echo '<meta property="og:title" content="' . esc_attr( get_the_title( $post->ID ) ) . '">' . "\n";
		echo '<meta property="og:description" content="' . esc_attr( $description ) . '">' . "\n";
		echo '<meta property="og:type" content="article">' . "\n";
		echo '<meta property="og:url" content="' . esc_url( get_permalink( $post->ID ) ) . '">' . "\n";
		if ( $thumbnail ) {
			echo '<meta property="og:image" content="' . esc_url( $thumbnail ) . '">' . "\n";
		}
		echo '<meta name="twitter:card" content="summary_large_image">' . "\n";
	}
}

/* ============================================================
   REGISTER CUSTOM FIELD: Article Featured Flag
   ============================================================ */
add_action( 'add_meta_boxes', function() {
	add_meta_box(
		'myrock_post_options',
		'Opciones MyRock',
		'myrock_post_options_meta_box',
		'post',
		'side',
		'default'
	);
} );

function myrock_post_options_meta_box( $post ) {
	wp_nonce_field( 'myrock_post_options', 'myrock_post_nonce' );
	$featured = get_post_meta( $post->ID, '_myrock_featured', true );
	$subtitle = get_post_meta( $post->ID, '_myrock_subtitle', true );
	?>
	<p>
		<label>
			<input type="checkbox" name="myrock_featured" value="1" <?php checked( $featured, '1' ); ?>>
			Artículo destacado
		</label>
	</p>
	<p>
		<label style="display:block;margin-bottom:4px;font-weight:600;">Subtítulo del artículo</label>
		<textarea name="myrock_subtitle" rows="2" style="width:100%;"><?php echo esc_textarea( $subtitle ); ?></textarea>
	</p>
	<?php
}

add_action( 'save_post', function( $post_id ) {
	if ( ! isset( $_POST['myrock_post_nonce'] ) ) return;
	if ( ! wp_verify_nonce( $_POST['myrock_post_nonce'], 'myrock_post_options' ) ) return;
	if ( defined( 'DOING_AUTOSAVE' ) && DOING_AUTOSAVE ) return;
	if ( ! current_user_can( 'edit_post', $post_id ) ) return;

	update_post_meta( $post_id, '_myrock_featured', isset( $_POST['myrock_featured'] ) ? '1' : '0' );
	update_post_meta( $post_id, '_myrock_subtitle', sanitize_textarea_field( $_POST['myrock_subtitle'] ?? '' ) );
} );
