# Inference Domain and Subregions
Heidi Rodenhizer

This document describes the process of creating the inference domain
over which RTS will be mapped in RTSMapping_v2. This domain is based on
the intersection between the Arctic boreal region, the permafrost
region, and the ArcticDEM v4 Mosaic. Additionally, it was split into a
northern and southern domain based on the extent of PlanetScope Global
Quarterly Visual Basemaps which are only guaranteed to 74 degrees N (but
actually extend to approximately 76 degrees N in most cases).

# Domain

## Dissolve Internal Boundaries of ArcticDEM and Planet Grids

### ArcticDEM

``` r
arcdem_domain = arctic_dem |>
  summarise(geometry = st_union(geometry))
```

![](inference_domain_files/figure-commonmark/unnamed-chunk-10-1.png)

### Planet Grids

``` r
# planet_domain = planet_grids |>
#   summarise(geometry = st_union(geometry)) # |>
```

![](inference_domain_files/figure-commonmark/unnamed-chunk-13-1.png)

## Prepare Permafrost Extent Map

The permafrost probability map is highly spotty. We are including any
areas that fall within the Arctic boreal region and have some coverage
of permafrost within a 100 x 100 km pixel.

``` r
perm_prob = perm_prob |>
  classify(
    rcl = matrix(
      c(0, 100, 1),
      ncol = 3,
      byrow = TRUE
    )
  ) |>
  as.factor() |>
  crop(arctic_boreal) |>
  mask(arctic_boreal)
levels(perm_prob) = data_frame(id = c(1), permafrost = c("Permafrost Possible"))
```

    Warning: `data_frame()` was deprecated in tibble 1.1.0.
    ℹ Please use `tibble()` instead.

``` r
perm_prob_sf = perm_prob |>
  aggregate(
    fact = 100,
    fun = any_ones
  ) |> # Check if there is any probability of permafrost within 100 km
  as.polygons() |>
  st_as_sf()
```

    <SpatRaster> resampled to 500650 cells.

![](inference_domain_files/figure-commonmark/unnamed-chunk-15-1.png)

## Circumpolar Domain

We started with the Arctic Boreal region that Anna V made to create our
circumpolar region. This region is first reduced by taking the
intersection with the ArcticDEM, because we need data from the
ArcticDEM. It is further reduced by taking the intersection with the
permafrost zone, because permafrost is a necessary precursor for RTS.

``` r
circumpolar = arctic_boreal |>
  st_intersection(arcdem_domain) |>
  st_intersection(perm_prob_sf)
```

    Warning: attribute variables are assumed to be spatially constant throughout
    all geometries

![](inference_domain_files/figure-commonmark/unnamed-chunk-17-1.png)

## Circumpolar South Domain

This is the portion of the ArcticDEM for which we have Planet imagery.

``` r
circumpolar_south = circumpolar |>
  st_intersection(planet_domain)
```

    Warning: attribute variables are assumed to be spatially constant throughout
    all geometries

![](inference_domain_files/figure-commonmark/unnamed-chunk-20-1.png)

## Circumpolar North Domain

This is the portion of the Circumpolar region for which we do not have
Planet data.

``` r
circumpolar_north = circumpolar |>
  st_difference(planet_domain)
```

    Warning: attribute variables are assumed to be spatially constant throughout
    all geometries

![](inference_domain_files/figure-commonmark/unnamed-chunk-23-1.png)

# Subregions

## Clip to Domain

The ecoregions from Dinerstein et al. 2017 need to be clipped to the
circumpolar region created above.

``` r
subregions = ecoregions |>
  filter(BIOME_NAME %in% c("Tundra", "Boreal Forests/Taiga")) |>
  st_intersection(circumpolar) |>
  st_as_sf() |>
  select(ECO_NAME, COLOR) |>
  arrange(ECO_NAME)
```

    Warning: attribute variables are assumed to be spatially constant throughout
    all geometries

![](inference_domain_files/figure-commonmark/unnamed-chunk-26-1.png)

## Check Training Polygon Counts by Subregion

``` r
region_poly_count = training_polys |>
  st_join(
    subregions |>
      select(RegionName = ECO_NAME),
    join = st_nearest_feature
  ) |>
  st_drop_geometry() |>
  summarise(
    RTSCount = n(),
    .by = RegionName
  ) |>
  mutate(
    RTSPercent = RTSCount / sum(RTSCount)
  )
```

![](inference_domain_files/figure-commonmark/unnamed-chunk-29-1.png)

![](inference_domain_files/figure-commonmark/unnamed-chunk-30-1.png)

``` r
small_clusters_percent = region_poly_count |>
  filter(RTSPercent < 0.1) |>
  summarise(TotalPercentSmallClusters = sum(RTSPercent))
small_clusters_percent
```

      TotalPercentSmallClusters
    1                 0.3434874
