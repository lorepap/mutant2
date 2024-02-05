import csv
import json

class Logger:
    def __init__(self, csv_file, columns):
        self.csv_file = csv_file
        self.csv_header_written = False
        self.cols = columns
        
    def log(self, data):
        # Append data to CSV file
        self.append_to_csv(data)

    def append_to_csv(self, data):
        # Check if the CSV file exists and write the header if not already written
        with open(self.csv_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            if not self.csv_header_written:
                # Write header only if it's the first time
                csv_writer.writerow(self.cols)  # Replace with your column names
                self.csv_header_written = True
            
            # Write the data to the CSV file
            csv_writer.writerow(data)

# Example usage:
# logger = Logger("example.log", "example.csv")
# logger.log("1,2,3")
