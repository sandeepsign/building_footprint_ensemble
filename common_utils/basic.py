from datetime import datetime

# Get date and time as a human-readable string (for filenames primarily)
def get_datetime_now():
    return datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

# Helper function to round a non-integer number to a set number of decimal places
def roundf(num, places):
    return round(num * 10**places) / 10**places
