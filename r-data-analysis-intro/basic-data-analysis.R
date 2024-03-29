
knitr::purl("basic-data-analysis.qmd")


#| column: page

library(readxl) # A useful library for reading Excel files

alamar_raw_plate <- read_excel(
  "jt58 etoposide kill curve 24h post eto wt vs 3ko13 3ko17 alamar.xlsx",
  range = "B15:M22", # For plate data, know where your plate reader puts it!
  col_names = FALSE
)

# Or, since we only use read_excel, we could just write `readxl::read_excel`
# (This is better practice than loading the entire library for one function)

alamar_raw <- alamar_raw_plate[1:7, 2:10] # Trim away empty wells
# Or, equivalently: alamar_raw <- alamar_raw_plate[-8, 2:10]

columns <- c(
  "WT_Par_1", "WT_Par_2", "WT_Par_3",
  "M3KO_13_1", "M3KO_13_2", "M3KO_13_3",
  "M3KO_17_1", "M3KO_17_2", "M3KO_17_3"
)

colnames(alamar_raw) <- columns

# Get and subtract the control wells

background <- as.numeric(alamar_raw[1,]) # `mean()` won't work on a data frame
avg_background <- mean(background)

alamar_corrected <- alamar_raw - avg_background # Subtracts from each well
alamar_corrected <- alamar_corrected[-1,] # Remove the control wells

# Set the concentrations and add to the table

etoposide_concs <- rev(c(0, 10^seq(0, 2, by = 0.5)))
alamar_corrected$conc <- etoposide_concs

# Print out a preview of the data frame
knitr::kable(alamar_corrected) # kable() gives us the table in a pretty format




alamar <- tidyr::pivot_longer(
  data = alamar_corrected,
  cols = -conc, # all columns except `conc`
  names_sep = "_",
  names_to = c("genotype", "clone", "replicate"),
  values_to = "corrected_absorbance"
)

knitr::kable(head(alamar, 10)) # Just show the first 10 rows



library(drc) # Both drm & LL.4() are from this library, so OK to import it all

dose_response_model <- drm(
  corrected_absorbance ~ conc, # Model absorbance as a function of concentration
  fct = LL.4(), # This says to use the 4-parameter logistic model
  curveid = genotype, # This says: do a separate model/curve for each genotype
  data = alamar # specifies the dataframe
)

ec50 <- ED(dose_response_model, 50) # (We can choose any %, not just 50%)

plot(dose_response_model)
abline(v=c(ec50[1],ec50[2]), col=c("blue", "red"), lty=c(1,3))



library(ggplot2)
library(ggprism)

# First set the genotype to a factor so we can control the order (for legends)
# (ggplot will convert it under the hood, and the default order is alphabetical)
alamar$genotype <- factor(alamar$genotype, levels = c("WT", "M3KO"))

# Then we make the labels useful for plotting
levels(alamar$genotype) <- c("Wild-type", "IFITM3 KO")

max_y <- max(alamar$corrected_absorbance)

plot <- ggplot(
  alamar,
  aes(x = conc, y = corrected_absorbance, color = genotype, fill = genotype)
)

plot <- plot +
  geom_line(
    stat = "smooth",
    method = drm,
    method.args = list(fct = LL.4()),
    linewidth = 2,
    se = FALSE
  ) +
  geom_point(
    color = "black",
    pch = 21,
    size = 3,
    alpha = 0.8,
    stroke = 1.2
  ) +
  scale_color_manual(values = c("#5278ec", "#bb0c40")) +
  scale_fill_manual(values = c("#5278ec", "#bb0c40")) +
  theme_prism(base_size = 14) +
  scale_y_continuous(limits = c(NA, max_y * 1.1)) +
  scale_x_continuous(
    trans = scales::pseudo_log_trans(0.1),
    labels = scales::label_number(drop0trailing = TRUE),
    breaks = c(0, 0.3, 1, 3, 10, 30, 100)
  ) +
  labs(x = "Etoposide (µM)", y = "Abs (450)")

# add the EC50 lines
plot <- plot +
  geom_vline(
    xintercept = ec50[1],
    color = "#5278ec",
    linewidth = 1.1,
    linetype = 2
  ) +
  geom_vline(
    xintercept = ec50[2],
    color = "#bb0c40",
    linewidth = 1.1,
    linetype = 3
  )

plot$theme[c("legend.text.align", "legend.title.align")] <- NULL

plot

