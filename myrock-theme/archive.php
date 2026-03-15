<?php
/**
 * Blog listing — archive.php
 * Handles the main blog index (/blog/) and category/tag archives.
 *
 * @package MyRock_Theme
 */

defined( 'ABSPATH' ) || exit;
?>
<!DOCTYPE html>
<html <?php language_attributes(); ?>>
<head>
  <meta charset="<?php bloginfo( 'charset' ); ?>">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <?php myrock_seo_meta(); ?>
  <?php wp_head(); ?>
</head>
<body <?php body_class( 'mr-archive' ); ?>>
<?php wp_body_open(); ?>

<?php get_template_part( 'template-parts/site-header' ); ?>

<main id="main" class="mr-main">

  <!-- Archive Header -->
  <header class="mr-archive-header">
    <div class="mr-container">
      <?php myrock_breadcrumbs(); ?>

      <?php if ( is_category() ) : ?>
        <div class="mr-archive-eyebrow">Categoría</div>
        <h1 class="mr-archive-title"><?php single_cat_title(); ?></h1>
        <?php if ( category_description() ) : ?>
          <p class="mr-archive-desc"><?php echo wp_kses_post( category_description() ); ?></p>
        <?php endif; ?>

      <?php elseif ( is_tag() ) : ?>
        <div class="mr-archive-eyebrow">Etiqueta</div>
        <h1 class="mr-archive-title"><?php single_tag_title(); ?></h1>

      <?php elseif ( is_search() ) : ?>
        <div class="mr-archive-eyebrow">Resultados</div>
        <h1 class="mr-archive-title">
          <?php echo esc_html( get_search_query() ) ?: 'Búsqueda'; ?>
        </h1>
        <p class="mr-archive-desc">
          <?php printf( '%d resultado%s encontrado%s', $wp_query->found_posts, $wp_query->found_posts !== 1 ? 's' : '', $wp_query->found_posts !== 1 ? 's' : '' ); ?>
        </p>

      <?php else : ?>
        <div class="mr-archive-eyebrow">Blog</div>
        <h1 class="mr-archive-title">Perspectiva,<br>estrategia y <em>criterio</em>.</h1>
        <p class="mr-archive-desc">
          Artículos de diseño, desarrollo y estrategia digital para marcas que quieren posicionarse donde muy pocos llegan.
        </p>
      <?php endif; ?>

      <!-- Search bar -->
      <form role="search" method="get" action="<?php echo esc_url( home_url( '/' ) ); ?>" class="mr-search-form">
        <input type="search" name="s" class="mr-search-input"
               placeholder="Buscar artículos…"
               value="<?php echo esc_attr( get_search_query() ); ?>"
               aria-label="Buscar">
        <button type="submit" class="mr-search-btn" aria-label="Buscar">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <circle cx="6.5" cy="6.5" r="5" stroke="currentColor" stroke-width="1.5"/>
            <path d="M10.5 10.5L14 14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </button>
      </form>
    </div>
  </header>

  <!-- Category Filter -->
  <?php if ( ! is_search() ) : ?>
  <div class="mr-category-nav">
    <div class="mr-container">
      <a href="<?php echo esc_url( get_post_type_archive_link( 'post' ) ); ?>"
         class="mr-cat-pill <?php echo ! is_category() && ! is_tag() ? 'active' : ''; ?>">
        Todos
      </a>
      <?php
      $cats = get_categories( [ 'hide_empty' => true, 'number' => 12 ] );
      foreach ( $cats as $cat ) :
        $active = is_category( $cat->term_id ) ? 'active' : '';
      ?>
        <a href="<?php echo esc_url( get_category_link( $cat->term_id ) ); ?>"
           class="mr-cat-pill <?php echo $active; ?>">
          <?php echo esc_html( $cat->name ); ?>
        </a>
      <?php endforeach; ?>
    </div>
  </div>
  <?php endif; ?>

  <!-- Posts Grid -->
  <section class="mr-posts-section">
    <div class="mr-container">

      <?php if ( have_posts() ) : ?>

        <div class="mr-posts-grid">
          <?php
          $is_first = true;
          while ( have_posts() ) :
            the_post();
            $featured_class = get_post_meta( get_the_ID(), '_myrock_featured', true ) === '1' ? 'mr-post-card--featured' : '';
            ?>
            <article id="post-<?php the_ID(); ?>" <?php post_class( 'mr-post-card ' . $featured_class ); ?>>

              <?php if ( has_post_thumbnail() ) : ?>
                <a href="<?php the_permalink(); ?>" class="mr-post-card__image-link" tabindex="-1" aria-hidden="true">
                  <div class="mr-post-card__image">
                    <?php the_post_thumbnail( 'myrock-card' ); ?>
                  </div>
                </a>
              <?php else : ?>
                <div class="mr-post-card__image mr-post-card__image--placeholder">
                  <span class="mr-post-card__placeholder-text"><?php echo esc_html( mb_substr( get_the_title(), 0, 2 ) ); ?></span>
                </div>
              <?php endif; ?>

              <div class="mr-post-card__body">

                <div class="mr-post-card__meta">
                  <?php
                  $cat = get_the_category();
                  if ( $cat ) :
                  ?>
                    <a href="<?php echo esc_url( get_category_link( $cat[0]->term_id ) ); ?>"
                       class="mr-post-card__cat">
                      <?php echo esc_html( $cat[0]->name ); ?>
                    </a>
                  <?php endif; ?>
                  <span class="mr-post-card__time"><?php echo esc_html( myrock_reading_time() ); ?></span>
                </div>

                <h2 class="mr-post-card__title">
                  <a href="<?php the_permalink(); ?>"><?php the_title(); ?></a>
                </h2>

                <?php
                $subtitle = get_post_meta( get_the_ID(), '_myrock_subtitle', true );
                if ( $subtitle ) : ?>
                  <p class="mr-post-card__subtitle"><?php echo esc_html( $subtitle ); ?></p>
                <?php else : ?>
                  <p class="mr-post-card__excerpt"><?php the_excerpt(); ?></p>
                <?php endif; ?>

                <div class="mr-post-card__footer">
                  <div class="mr-post-card__author">
                    <?php echo get_avatar( get_the_author_meta( 'ID' ), 28, '', '', [ 'class' => 'mr-post-card__avatar' ] ); ?>
                    <span class="mr-post-card__author-name"><?php the_author(); ?></span>
                  </div>
                  <time class="mr-post-card__date" datetime="<?php echo get_the_date( 'c' ); ?>">
                    <?php echo get_the_date( 'j M Y' ); ?>
                  </time>
                </div>

              </div>
            </article>
          <?php endwhile; ?>
        </div>

        <!-- Pagination -->
        <nav class="mr-pagination" aria-label="Paginación">
          <?php
          echo paginate_links( [
            'prev_text' => '← Anterior',
            'next_text' => 'Siguiente →',
            'type'      => 'list',
          ] );
          ?>
        </nav>

      <?php else : ?>

        <div class="mr-no-posts">
          <p class="mr-no-posts__text">No se encontraron artículos.</p>
          <a href="<?php echo esc_url( home_url( '/blog/' ) ); ?>" class="mr-btn-ghost">
            Ver todos los artículos →
          </a>
        </div>

      <?php endif; ?>
    </div>
  </section>

</main>

<?php get_template_part( 'template-parts/blog-newsletter-cta' ); ?>
<?php get_template_part( 'template-parts/site-footer' ); ?>

<?php wp_footer(); ?>
</body>
</html>
