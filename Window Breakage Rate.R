################################################################################
# A simple example integrating descriptive, predictive, and prescriptive
# analytics into a shiny app
# Team & Member Names:Team 8 Hanbing Yang, Yuchen Zhang, Pratik Mahesh Merchant, Yi-chun Huang
################################################################################
# https://shiny.posit.co/
# https://rstudio.github.io/shinythemes/
#install.packages("shinythemes")
library(shiny)
library(shinythemes)

ui <- fluidPage(
  theme = shinytheme("flatly"),
   
  # Add a dark background for better content readability
  tags$style(HTML("
    /* Adjusting text color for h3 and p elements under the Introduction tab */
    .tab-content .active h3, .tab-content .active p {
         color: #000000;  /* Black color */
    }
    
    /* Optional: Adding a slightly darker background for better contrast */
    .tab-content .active {
        background-color: #FFFFFF;  /* White background */
        padding: 15px;  /* Some padding for better spacing */
        border-radius: 5px;  /* Rounded corners */
    .container {
    max-width: 1200px;  /* Adjust the width as needed */
   
    }
  /* Adjusting text color for h3 and p elements under the Predictive Analytics tab */
    .tab-content .active .tab-pane h3, .tab-content .active .tab-pane p {
         color: #000000;  /* Black color */
    }

    /* Adjusting background color for the Predictive Analytics tab */
    .nav-tabs .active a[href='#tab-predictive'] {
        background-color: #FFFFFF;  /* White background */
    }
    
    /* Optional: Adding a slightly darker background for better contrast */
    .tab-content .active .tab-pane {
        background-color: #FFFFFF;  /* White background */
        padding: 15px;  /* Some padding for better spacing */
        border-radius: 5px;  /* Rounded corners */
    }
    
    .container {
        max-width: 1200px;  /* Adjust the width as needed */
    }
  
    }
")),
  
  titlePanel("Using R for Analytics"),
  mainPanel(
    navbarPage(
      tabPanel("--"),
      tabPanel("Introduction",
               h3("Business Problem:"),
               p("Background:
In the window manufacturing industry, a significant challenge faced by producers is the breakage rate during the manufacturing process. High breakage rates can lead to increased production costs, resource wastage, potential delays in order fulfillment, and customer dissatisfaction."),
               p("Objective:
The primary goal is to leverage advanced data analytics to provide insights into the manufacturing process and supply chain. By doing so, we aim to identify:"),
               p("Specific manufacturing process settings that could be optimized.
Suppliers that consistently provide better quality materials leading to lower breakage rates.
Customer window specifications that are more prone to breakages, to better manage expectations or improve production techniques for those specifications."),
               p("Proposed Solution:
Develop a Decision Support System (DSS) that integrates descriptive, predictive, and prescriptive analytics. This system will be designed to:"),
               
               p("Descriptive Analytics: Provide a comprehensive overview of the breakage rates, correlating them with different variables such as window size, glass thickness, ambient temperature during manufacturing, cut speed, etc."),
               
               p("Predictive Analytics: Forecast potential breakage rates based on proposed changes in manufacturing settings or supplier choices."),
               
               p("Prescriptive Analytics: Offer actionable recommendations to technicians on how to adjust manufacturing settings, choose suppliers, or handle specific customer window specifications to minimize breakages."),
               
               p("End Users: The primary users of this DSS will be window manufacturing technicians. The interactive nature of the DSS will empower these technicians with real-time data analytics insights, enabling them to make informed decisions swiftly.:"),
               
               p("Significance:
Reducing window breakage rates not only has direct cost-saving implications but also enhances the brand's reputation, ensures timely deliveries, and boosts customer satisfaction. By offering technicians a tool that integrates various analytics methods, we ensure a holistic approach to problem-solving in the window manufacturing domain."),
               
               h3("Analytics Problem Framing"),
               p("From the dataset, it appears that one potential problem is predicting the Breakage Rate based on various parameters like window size, glass thickness, ambient temperature, etc. This is a regression-type problem. Thus, we can use regression algorithms (e.g., linear regression) to model and predict the Breakage Rate. We can use various libraries, including caret, for this purpose.
"),
               p("We can hypothesize certain relationships based on domain knowledge and then validate them using the data. For instance, one might expect that:
"),
               p("As the glass thickness increases, the breakage rate might decrease since thicker glass may be more robust.
Ambient temperature might influence the breakage rate, with extreme temperatures possibly leading to more breakages.
Different window types (Vinyl, Aluminum, Wood) might have varying breakage rates due to their inherent properties."),
               p("Some assumptions could include:"),
               p("Holding all others the same, the larger the window size is, the higher the window breakage rate will be."),
               p("Holding all others the same, the thicker the window is, the lower the window breakage rate will be."),
               p("Holding all others the same, the higher the ambient temperature is, the lower the window breakage rate will be."),
               p("Holding all others the same, the higher the cut speed is, the higher the window breakage rate will be."),
               p("Holding all others the same, the higher the edge deletion rate is, the higher the window breakage rate will be."),
               p("Holding all others the same, the higher the silicon viscosity is, the higher the window breakage rate will be."),
               p("Holding all others the same, the window type has a significant impact on the window breakage rate."),
               
               p("Key metrics of success:"),
               p("For a regression problem, typical metrics of success include:"),
               p("Root Mean Square Error (RMSE): Indicates the square root of the average squared differences between predicted and actual observations. It gives more weight to larger errors."),
               p("R-squared: Represents the proportion of the variance for the dependent variable that's explained by independent variables in a regression model."),
              
               
               tableOutput("dataDictTable")
      ),
      
      
      tabPanel("Descriptive Analytics",
               plotOutput(outputId="multi_plot", height="800px", width="1000px")
               
      ),
      tabPanel("Predictive Analytics",
               mainPanel(
                 fluidRow(
                   column(8, tableOutput(outputId="coeffTable"), style = "padding-right: 20px;"),      
                   column(4, tableOutput(outputId="metricsTable"), style = "padding-left: 20px;")
                 ),
               )
      ),

      
      tabPanel("Prescriptive Analytics",
               p("Objective function:"),
               p("Min BreakageRate = 262.44 - 0.18 * WindowSize - 522.8 * Glassthickness +"),
               p("0.12 * AmbientTemp + 0.47 * Cutspeed + 0.82 * EdgeDeletionrate + 0.09 * Siliconviscosity - "),
               p("0.11 * LocationIowa + 1.08 * LocationMichigan + 0.49 * Location_Minnesota +"),
               p("1.17 * TypeAluminum + 1.01 * TypeVinyl"),
               sidebarPanel(
                 # Slider for setting Window Size (a non-controllable decision variable)
                 sliderInput(inputId = "windowSize", 
                             label = "Window Size", 
                             min = 51.91,   # Assuming a reasonable minimum window size
                             max = 75.56,  # Assuming a reasonable maximum window size
                             value = 70, # Default window size 
                             step = 0.01    # Step size for incremental changes
                 ),
                 # Slider for setting Ambient Temperature (a non-controllable decision variable)
                 sliderInput(inputId = "ambientTemp", 
                             label = "Ambient Temperature", 
                             min = 8.4,   # Minimum temperature value 
                             max = 24.09,  # Maximum temperature value
                             value = 15, # Default temperature value 
                             step = 0.01  # Step size for incremental changes
                 ),
                 # Dropdown select input for Window Type
                 selectInput(inputId = "windowType",
                             label = "Window Type",
                             choices = c("Vinyl", "Wood", "Aluminum"), # Add other types if needed
                             selected = "Vinyl" # Default selected value
                 ),
                 actionButton(inputId = "run2", label = "Run")
               ),
               textOutput(outputId="optResult"), 
               textOutput(outputId="optDecisions"),
               tableOutput("my_list")
               #fluidRow(
                # column(6,
               #         textOutput("myTableValues")
                # )
               #)
      )
    ),
    
   
  
  ),
)
server = function(input, output) {
  ################# Data Dictionary #######################################
  # Create the data dictionary dataframe in R
  
  data_dictionary <- data.frame(
    Variable_Name = c(
      "Breakage Rate", 
      "Window Size", 
      "Glass thickness", 
      "Ambient Temp", 
      "Cut speed", 
      "Edge Deletion rate", 
      "Spacer Distance", 
      "Window color", 
      "Window Type", 
      "Glass Supplier", 
      "Silicon Viscosity", 
      "Glass Supplier Location"
    ),
    R_Data_Type = c(
      "numeric", 
      "numeric", 
      "numeric", 
      "numeric", 
      "numeric", 
      "numeric", 
      "numeric", 
      "numeric", 
      "factor", 
      "factor", 
      "numeric", 
      "factor"
    ),
    Short_Variable_Description = c(
      "Rate at which windows break during manufacturing",
      "Size of the manufactured window",
      "Thickness of the glass used in the window",
      "Ambient temperature during manufacturing",
      "Speed at which the glass is cut",
      "Rate at which edges are deleted during manufacturing",
      "Distance between the spacers in the window",
      "Color intensity of the window",
      "Type of the window (e.g., Vinyl, Aluminum, Wood)",
      "Supplier of the glass",
      "Viscosity of the silicon used",
      "Location of the glass supplier"
    )
  )
  
  
  # Render the data dictionary as a table
  output$dataDictTable <- renderTable({
    data_dictionary
  })
  
  
  #################  Data loading #######################################
  library("xlsx")
  d<-read.xlsx(file="Window_Manufacturing.xlsx",1,header=T)
  names(d)[1] <- c("y")
  names(d) <- gsub("\\.", "", names(d))  
  d$WindowType <- as.factor(d$WindowType)
  d$GlassSupplier <- as.factor(d$GlassSupplier)
  str(d)
  d$GlassSupplierLocation <- as.factor(d$GlassSupplierLocation)
  
  calculate_na <- function(x) {
    sum(is.na(x))
  }
  
  na_col <- apply(d, 2, calculate_na)
  print(na_col)
  
  library(mice)
  imputedValues <- mice(data=d
                        , seed=2016     
                        , method="cart" 
                        , m=1           
                        , maxit = 1     
  )
  d <- mice::complete(imputedValues,1) 
  
  has_na <- FALSE
  
  # Loop through columns
  for (col in names(d)) {
    if (any(is.na(d[[col]]))) {
      cat("Column", col, "has missing values (NA)\n")
      has_na <- TRUE
    }
  }
  
  if (!has_na) {
    cat("No missing values found in any column.\n")
  }
  
  
  orgd <- d 
 
    ################# Descriptive Tab #######################################
  library(sqldf)
  result <- sqldf("SELECT WindowType, avg(y) FROM orgd group by WindowType")
  print(result)
  
    #install.packages('gridExtra')
  library(ggplot2)
  library(gridExtra)
  library(dplyr)
  output$multi_plot <- renderPlot({
    
    
    #A bar chart showing the average Breakage Rate for each Glass Supplier.# Bar chart
    plot2 <- ggplot(orgd, aes(x = GlassSupplier, y = y)) +
      stat_summary(fun = mean,
                   geom = "bar",
                   fill = "skyblue") +
      labs(title = "Average Breakage Rate for Each Glass Supplier",
           x = "Glass Supplier",
           y = "Average Breakage Rate") +
      theme_minimal()
    print(plot2)
    
    plot3 <-
      ggplot(orgd,
             aes(x = GlassSupplierLocation, y = y, fill = GlassSupplierLocation)) +
      geom_violin() +
      labs(title = "Distribution of Breakage Rate for Different Glass Supplier Locations",
           x = "Glass Supplier Location",
           y = "Breakage Rate") +
      theme_minimal()
    print(plot3)
    
    plot4 <- ggplot(orgd, aes(x = Cutspeed, y = y)) +
      geom_point(color = "#69b3a2", alpha = 0.7) +  geom_smooth(
        method = lm ,
        color = "red",
        fill = "#69b3a2",
        se = TRUE
      ) +
      labs(
        title = "Breakage Rate vs. Cut Speed",
        x = "Cut Speed",
        y = "Breakage Rate",
        color = "Window Type"
      ) +
      theme_minimal()
    print(plot4)
    
    # 5. Histogram for the Breakage Rate
    plot5 <- ggplot(orgd, aes(x = y)) + 
      geom_histogram(binwidth = 1, fill = "#69b3a2", color = "#e9ecef", alpha = 0.7) +
      ggtitle("Histogram of Breakage Rate")
    print(plot5)
    
    # 9. Box plot showing the Breakage Rate for each Glass Supplier Location
    plot9 <- ggplot(orgd, aes(x = GlassSupplierLocation, y = y)) + 
      geom_boxplot() +
      labs(title = "Box plot: Breakage Rate by Glass Supplier Location",
           x = "Glass Supplier Location",
           y = "Breakage Rate") +
      theme_minimal()
    print(plot9)
    
    # 10.
    plot10 <- ggplot(orgd,aes(x=Glassthickness,y=y)) + geom_point() +
      geom_smooth(method=lm , color="red", fill="#69b3a2", se=TRUE) 
    print(plot10)
    
    #11.
    plot11 <- ggplot(orgd, aes(x = WindowType, y = y)) +
      geom_boxplot() +
      labs(title = "Distribution of Breakage Rate for Different Window Types",
           x = "Window Type",
           y = "Breakage Rate") +
      theme_minimal()
    print(plot11)
    
    plot12 <- ggplot(orgd, aes(x = WindowType, y = y)) +
      stat_summary(fun = mean,
                   geom = "bar",
                   fill = "skyblue") +
      labs(title = "Average Breakage Rate for Each Window Type",
           x = "Window Type",
           y = "Average Breakage Rate") +
      theme_minimal()
    print(plot12)
    
    # Arrange in 2x2 grid
    grid.arrange(plot4,plot5,plot9,plot3,plot2,plot10,plot11,plot12,ncol=2)
    
  })
  
  #################  Data pre-processing #######################################
  library(caret)
  y <- d$y
  dummies <- dummyVars(y ~ ., data = d)
  ex <- data.frame(predict(dummies, newdata = d))
  names(ex) <- gsub("\\.", "", names(ex))
  d <- cbind(d$y, ex)
  rm(dummies)
 
  descrCor <-  cor(d[,2:ncol(d)])                           
  highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .85) 
  #summary(descrCor[upper.tri(descrCor)])                  
  highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.85)  
  filteredDescr <- d[,2:ncol(d)][,-highlyCorDescr] 
  descrCor2 <- cor(filteredDescr)                  
  #summary(descrCor2[upper.tri(descrCor2)])
  d <- cbind(t=y, filteredDescr)
  rm(filteredDescr, descrCor, descrCor2, highCorr, highlyCorDescr)
  
  d <- cbind(rep(1, nrow(d)), d[2:ncol(d)])  
  names(d)[1] <- "ones"
  comboInfo <- findLinearCombos(d)
  d <- d[, -comboInfo$remove]
  d <- d[, c(2:ncol(d))]
  d <- cbind(y, d)
  rm(y, comboInfo)
  
  nzv <- nearZeroVar(d, saveMetrics = TRUE)
  head(nzv)
  d <- d[, c(TRUE,!nzv$zeroVar[2:ncol(d)])]
  
  set.seed(1234)
  inTrain <- createDataPartition(y = d$y, p = .70, list = FALSE)
  train <- d[inTrain, ]
  test <- d[-inTrain, ]
  
  ################# Predictive Tab  #######################################
  
  library(caret)
  
  output$coeffTable <- renderTable({
    m1 <- lm(y ~ WindowSize+Glassthickness+AmbientTemp+Cutspeed
             +EdgeDeletionrate+SiliconViscosity
             +GlassSupplierLocationIowa+GlassSupplierLocationMichigan+GlassSupplierLocationMinnesota 
             + WindowTypeAluminum +WindowTypeVinyl, data=train)
    coeff_names <- rownames(summary(m1)[["coefficients"]])
    coeff_estimates <- summary(m1)[["coefficients"]][, "Estimate"]
    coeff_pvalues <- summary(m1)[["coefficients"]][, "Pr(>|t|)"]
    data.frame(Coefficient = coeff_names, Estimate = coeff_estimates, P_Value = coeff_pvalues)
  })
  
  
  output$metricsTable <- renderTable({
    # Predict on training set
    m1 <- lm(y ~WindowSize+Glassthickness+AmbientTemp+Cutspeed
             +EdgeDeletionrate+SiliconViscosity
             +GlassSupplierLocationIowa+GlassSupplierLocationMichigan+GlassSupplierLocationMinnesota 
             + WindowTypeAluminum +WindowTypeVinyl, data=train)
    train_pred <- predict(m1, newdata=train)
    train_res <- defaultSummary(data.frame(obs = train$y, pred = train_pred))
    
    # Predict on test set
    test_pred <- predict(m1, newdata=test)
    test_res <- defaultSummary(data.frame(obs = test$y, pred = test_pred))
    
    metrics_df <- data.frame(
      Dataset = c("Train", "Test"),
      RMSE = c(train_res["RMSE"], test_res["RMSE"]),
      #MAE = c(train_res["mae"], test_res["mae"]),
      R_squared = c(train_res["Rsquared"], test_res["Rsquared"])
      )
    
    
  })
  
  
  
  ################# Prescriptive Tab #######################################
  source("constraint.R")
  cons <- list()
  num = 1
  for (i in c("WindowSize","Glassthickness","AmbientTemp","Cutspeed","EdgeDeletionrate","SiliconViscosity")){
    cons[[num]] <- const(a=i,data=d)
    num <- num+1
  }
  print(cons)
  
    model2 <- eventReactive(input$run2, {

      library(lpSolveAPI)
      library(lpSolve)
      # there are two decision variables
      (lps.model <- make.lp(nrow=0, ncol=12))
      # real decision variables
      set.type(lps.model, columns=1, type="real")
      set.type(lps.model, columns=2, type="real")
      set.type(lps.model, columns=3, type="real")
      set.type(lps.model, columns=4, type="real")
      set.type(lps.model, columns=5, type="real")
      set.type(lps.model, columns=6, type="real")
      set.type(lps.model, columns=7, type="real")
      set.type(lps.model, columns=8, type="binary")
      set.type(lps.model, columns=9, type="binary")
      set.type(lps.model, columns=10, type="binary")
      set.type(lps.model, columns=11, type="binary")
      set.type(lps.model, columns=12, type="binary")
      
      
      # set objective function
      lp.control(lps.model, sense="min")
      set.objfn(lps.model, obj=c(1,-0.18,-522.80126,0.12292,0.47280, 0.81642,0.08902, 
                                 -0.11307, 1.07944, 0.48895,1.17419,1.01271))
      
      add.constraint(lps.model, c(1,0,0,0,0,0,0,0,0,0,0,0), "=", 262.44)
      add.constraint(lps.model, c(0,1,0,0,0,0,0,0,0,0,0,0), "=", input$windowSize)      #box constraint
 #     add.constraint(lps.model, c(0,1,0,0,0,0,0,0,0,0,0,0), "<=", 75.568)     #box constraint
      add.constraint(lps.model, c(0,0,1,0,0,0,0,0,0,0,0,0), ">=", 0.4905)      #box constraint
      add.constraint(lps.model, c(0,0,1,0,0,0,0,0,0,0,0,0), "<=", 0.5149)     #box constraint
      add.constraint(lps.model, c(0,0,0,1,0,0,0,0,0,0,0,0), "=",input$ambientTemp)
 #     add.constraint(lps.model, c(0,0,0,1,0,0,0,0,0,0,0,0), ">=", 8.3926)  #cost constraint
 #     add.constraint(lps.model, c(0,0,0,1,0,0,0,0,0,0,0,0), "<=", 24.0983)      #box constraint
      add.constraint(lps.model, c(0,0,0,0,1,0,0,0,0,0,0,0), ">=", 0.2964)  #cost constraint
      add.constraint(lps.model, c(0,0,0,0,1,0,0,0,0,0,0,0), "<=", 3.21574) 
      add.constraint(lps.model, c(0,0,0,0,0,1,0,0,0,0,0,0), ">=", 13.752)     #box constraint
      add.constraint(lps.model, c(0,0,0,0,0,1,0,0,0,0,0,0), "<=", 17.743) 
      add.constraint(lps.model, c(0,0,0,0,0,0,1,0,0,0,0,0), ">=", 7.8055)      #box constraint
      add.constraint(lps.model, c(0,0,0,0,0,0,1,0,0,0,0,0), "<=", 16.1960)     #box constraint
      add.constraint(lps.model, c(0,0,0,0,0,0,0,1,0,0,0,0), ">=", 0) 
      add.constraint(lps.model, c(0,0,0,0,0,0,0,1,0,0,0,0), "<=", 1)      #box constraint
      add.constraint(lps.model, c(0,0,0,0,0,0,0,0,1,0,0,0), ">=", 0) 
      add.constraint(lps.model, c(0,0,0,0,0,0,0,0,1,0,0,0), "<=", 1) 
      add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,1,0,0), ">=", 0) 
      add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,1,0,0), "<=", 1) 
     # add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,1,0), ">=", 0)   #al
     # add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,1,0), "<=", 1)   #al
     # add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,0,1), ">=", 0)  #vin
     # add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,0,1), "<=", 1)  #vin
     # add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,1,1), "<=", 1)
      add.constraint(lps.model, c(0,0,0,0,0,0,0,1,1,1,0,0), "<=", 1)
      
      # Add constraints for Window Type
      if (input$windowType == "Aluminum") {
        add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,1,0), "=", 1)
        add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,0,1), "=", 0)
      } else if (input$windowType == "Vinyl") {
        add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,1,0), "=", 0)
        add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,0,1), "=", 1)
      } else if (input$windowType == "Wood") {
        add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,1,0), "=", 0)
        add.constraint(lps.model, c(0,0,0,0,0,0,0,0,0,0,0,1), "=", 0)
      }
    # final model that is ready to be solved
    lps.model 
  })
 
 
  
  output$optResult <- renderText({
    solve(model2())
    paste("Optimal Solution:",round(get.objective(model2()),13))
  }) 
  
  
  result <- eventReactive(input$run2, {

      variable = c("Intercept", "WindowSize","Glassthickness", "AmbientTemp", "Cutspeed", "EdgeDeletionrate", "Siliconviscosity", 
                   "LocationIowa", "LocationMichigan", "LocationMinnesota", "TypeAluminum", "TypeVinyl")
      value <- as.character(unname(unlist(data.frame(get.variables(model2())))))
      data.frame(variable = variable, result = value)
 
    
  })
  
  output$my_list <- renderTable({
    result()
  })
  

}  


shinyApp(ui = ui, server = server)
