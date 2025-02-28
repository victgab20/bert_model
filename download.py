import gdown

file_id = "1dDAcKxmwi3QvbJeSr9wNOEAUlWTit9GY"
url = f"https://drive.google.com/uc?id={file_id}"
output = "dataset/balanced_train_filmes.csv"
gdown.download(url, output, quiet=False)


file_id1 = "1JrsI7nobdwcqr_0c2-KTVIaJKNvdepBe"
url1 = f"https://drive.google.com/uc?id={file_id1}"
output1 = "dataset/balanced_test_filmes.csv"
gdown.download(url1, output1, quiet=False)

file_id2 = "1aZotbHRKlKGT9TAPuDBEJLxnOBj9SnLI"
url2 = f"https://drive.google.com/uc?id={file_id2}"
output2 = "dataset/balanced_test_filmes.csv"
gdown.download(url2, output2, quiet=False)