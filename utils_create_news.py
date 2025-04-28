import pickle
import io


def try_load(file_path:str, file_name:str, client_s3, BUCKET_NAME='graduate'):
    try:
        with open(file_path+file_name, "rb") as file:
            return pickle.load(file)
    except:
        print('Load pdf from s3: original news')

        client_s3.fget_object(
            bucket_name=BUCKET_NAME,
            object_name=file_name,
            file_path=file_path+file_name
            )
        
        with open(file_path+file_name, "rb") as file:
            return pickle.load(file)


def save_s3(pickle_data, object_key, client_s3, BUCKET_NAME='graduate'):

    pickle_data = pickle.dumps(pickle_data)

    client_s3.put_object(
        bucket_name=BUCKET_NAME, 
        object_name=object_key, 
        data=io.BytesIO(pickle_data), 
        length=len(pickle_data), 
        content_type="application/octet-stream"
        )