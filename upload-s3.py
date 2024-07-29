import supabase

URL = "https://caiimowdcnrwheoelnfg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNhaWltb3dkY25yd2hlb2VsbmZnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyMTY1MTM0OCwiZXhwIjoyMDM3MjI3MzQ4fQ.CfPM0xL4K1H8pgmmBiFUv0r2Q4OIQv88IL0br0ytn08"

supabase_client = supabase.create_client(URL, SUPABASE_KEY)


def upload_data():
    bucket = supabase_client.storage.get_bucket("data")
    bucket.upload("data.csv", "data.csv")
    print("Data uploaded to Supabase Storage")


upload_data()