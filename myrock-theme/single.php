<?php
/**
 * Single post template — single.php
 *
 * @package MyRock_Theme
 */

defined( 'ABSPATH' ) || exit;

the_post();

$subtitle    = get_post_meta( get_the_ID(), '_myrock_subtitle', true );
$categories  = get_the_category();
$tags        = get_the_tags();
$related     = myrock_related_posts( get_the_ID(), 3 );
?>
<!DOCTYPE html>
<html <?php language_attributes(); ?>>
<head>
  <meta charset="<?php bloginfo( 'charset' ); ?>">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <?php myrock_seo_meta(); ?>
  <?php wp_head(); ?>
</head>
<body <?php body_class( 'mr-single' ); ?>>
<?php wp_body_open(); ?>

<?php get_template_part( 'template-parts/site-header' ); ?>

<main id="main" class="mr-main">

  <!-- Post Header -->
  <header class="mr-post-header">
    <div class="mr-container mr-container--narrow">

      <?php myrock_breadcrumbs(); ?>

      <!-- Category -->
      <?php if ( $categories ) : ?>
        <a href="<?php echo esc_url( get_category_link( $categories[0]->term_id ) ); ?>"
           class="mr-post-header__cat">
          <?php echo esc_html( $categories[0]->name ); ?>
        </a>
      <?php endif; ?>

      <!-- Title -->
      <h1 class="mr-post-header__title"><?php the_title(); ?></h1>

      <!-- Subtitle -->
      <?php if ( $subtitle ) : ?>
        <p class="mr-post-header__subtitle"><?php echo esc_html( $subtitle ); ?></p>
      <?php endif; ?>

      <!-- Meta bar -->
      <div class="mr-post-header__meta">
        <div class="mr-post-header__author">
          <?php echo get_avatar( get_the_author_meta( 'ID' ), 36, '', '', [ 'class' => 'mr-post-header__avatar' ] ); ?>
          <div class="mr-post-header__author-info">
            <span class="mr-post-header__author-name"><?php the_author(); ?></span>
            <span class="mr-post-header__author-role">MyRock</span>
          </div>
        </div>
        <div class="mr-post-header__details">
          <time datetime="<?php echo get_the_date( 'c' ); ?>" class="mr-post-header__date">
            <?php echo get_the_date( 'j \d\e F, Y' ); ?>
          </time>
          <span class="mr-post-header__dot" aria-hidden="true">·</span>
          <span class="mr-post-header__read-time"><?php echo esc_html( myrock_reading_time() ); ?></span>
        </div>
      </div>

    </div>
  </header>

  <!-- Hero Image -->
  <?php if ( has_post_thumbnail() ) : ?>
    <div class="mr-post-hero">
      <div class="mr-container">
        <?php the_post_thumbnail( 'myrock-hero', [ 'class' => 'mr-post-hero__img', 'loading' => 'eager' ] ); ?>
      </div>
    </div>
  <?php endif; ?>

  <!-- Post Content -->
  <div class="mr-post-content-wrap">
    <div class="mr-container mr-container--narrow">

      <!-- Table of Contents (auto-generated via JS) -->
      <div id="mr-toc" class="mr-toc" aria-label="Contenido"></div>

      <div class="mr-post-content">
        <?php the_content(); ?>
      </div>

      <!-- Tags -->
      <?php if ( $tags ) : ?>
        <div class="mr-post-tags">
          <?php foreach ( $tags as $tag ) : ?>
            <a href="<?php echo esc_url( get_tag_link( $tag->term_id ) ); ?>" class="mr-tag-pill">
              #<?php echo esc_html( $tag->name ); ?>
            </a>
          <?php endforeach; ?>
        </div>
      <?php endif; ?>

      <!-- Author box -->
      <div class="mr-author-box">
        <?php echo get_avatar( get_the_author_meta( 'ID' ), 64, '', '', [ 'class' => 'mr-author-box__avatar' ] ); ?>
        <div class="mr-author-box__body">
          <div class="mr-author-box__label">Escrito por</div>
          <div class="mr-author-box__name"><?php the_author(); ?></div>
          <?php $bio = get_the_author_meta( 'description' ); ?>
          <?php if ( $bio ) : ?>
            <p class="mr-author-box__bio"><?php echo esc_html( $bio ); ?></p>
          <?php endif; ?>
        </div>
      </div>

      <!-- Post Navigation -->
      <nav class="mr-post-nav" aria-label="Artículos">
        <?php
        $prev = get_previous_post();
        $next = get_next_post();
        ?>
        <?php if ( $prev ) : ?>
          <a href="<?php echo esc_url( get_permalink( $prev->ID ) ); ?>" class="mr-post-nav__link mr-post-nav__link--prev">
            <span class="mr-post-nav__dir">← Anterior</span>
            <span class="mr-post-nav__title"><?php echo esc_html( get_the_title( $prev->ID ) ); ?></span>
          </a>
        <?php endif; ?>
        <?php if ( $next ) : ?>
          <a href="<?php echo esc_url( get_permalink( $next->ID ) ); ?>" class="mr-post-nav__link mr-post-nav__link--next">
            <span class="mr-post-nav__dir">Siguiente →</span>
            <span class="mr-post-nav__title"><?php echo esc_html( get_the_title( $next->ID ) ); ?></span>
          </a>
        <?php endif; ?>
      </nav>

    </div>
  </div>

  <!-- Post CTA -->
  <?php get_template_part( 'template-parts/post-cta' ); ?>

  <!-- Related Posts -->
  <?php if ( $related ) : ?>
    <section class="mr-related">
      <div class="mr-container">
        <h2 class="mr-related__title">También puede interesarte</h2>
        <div class="mr-related__grid">
          <?php foreach ( $related as $rel_post ) : setup_postdata( $rel_post ); ?>
            <article class="mr-related__card">
              <?php if ( has_post_thumbnail( $rel_post->ID ) ) : ?>
                <a href="<?php echo esc_url( get_permalink( $rel_post->ID ) ); ?>" class="mr-related__img-link">
                  <?php echo get_the_post_thumbnail( $rel_post->ID, 'myrock-thumb', [ 'class' => 'mr-related__img' ] ); ?>
                </a>
              <?php endif; ?>
              <div class="mr-related__body">
                <?php $rc = get_the_category( $rel_post->ID ); ?>
                <?php if ( $rc ) : ?>
                  <span class="mr-related__cat"><?php echo esc_html( $rc[0]->name ); ?></span>
                <?php endif; ?>
                <h3 class="mr-related__post-title">
                  <a href="<?php echo esc_url( get_permalink( $rel_post->ID ) ); ?>">
                    <?php echo esc_html( get_the_title( $rel_post->ID ) ); ?>
                  </a>
                </h3>
                <div class="mr-related__meta">
                  <time><?php echo get_the_date( 'j M Y', $rel_post->ID ); ?></time>
                  <span>·</span>
                  <span><?php echo esc_html( myrock_reading_time( $rel_post->ID ) ); ?></span>
                </div>
              </div>
            </article>
          <?php endforeach; wp_reset_postdata(); ?>
        </div>
      </div>
    </section>
  <?php endif; ?>

</main>

<?php get_template_part( 'template-parts/blog-newsletter-cta' ); ?>
<?php get_template_part( 'template-parts/site-footer' ); ?>

<script>
// Auto Table of Contents
(function() {
  const content = document.querySelector('.mr-post-content');
  const toc     = document.getElementById('mr-toc');
  if (!content || !toc) return;

  const headings = content.querySelectorAll('h2, h3');
  if (headings.length < 3) return;

  let html = '<div class="mr-toc__label">Contenido</div><ol class="mr-toc__list">';
  headings.forEach((h, i) => {
    const id = 'mr-heading-' + i;
    h.id = id;
    const level = h.tagName === 'H3' ? 'mr-toc__item--sub' : '';
    html += `<li class="mr-toc__item ${level}"><a href="#${id}" class="mr-toc__link">${h.textContent}</a></li>`;
  });
  html += '</ol>';
  toc.innerHTML = html;
  toc.style.display = 'block';
})();
</script>

<?php wp_footer(); ?>
</body>
</html>
