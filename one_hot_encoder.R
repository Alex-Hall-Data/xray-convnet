#script to change the classifications to one hot encoding

xray_data <- read.csv("C:\\Users\\alex.hall\\Documents\\datasets\\xray\\Data_Entry_2017.csv\\Data_Entry_2017.csv")

labels_list <- paste(xray_data$Finding.Labels,collapse="|")%>%
  strsplit(.,"\\|")

unique_labels<-unique(as.vector(labels_list)[[1]])

#initialise empty list of vectors (zeros)
labels<-list()
for (i in(1:nrow(xray_data))){
  labels[i]<-list(rep(0,length(unique_labels)))
}

#make list of k hot encoded vectors
for(i in (1:length(labels))){
  for(j in (1: length(labels[[i]]))){
    labels[[i]][j]<- ifelse(grepl(unique_labels[j],xray_data$Finding.Labels[i]),1,0)
  }
  print(i)
}