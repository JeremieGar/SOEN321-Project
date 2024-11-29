import requests
from bs4 import BeautifulSoup


def fetch_policy(url):
    """
    Fetches and extracts the privacy policy text from a given URL.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Look for elements that contain policy text (adjust based on structure)
    paragraphs = soup.find_all('p')  # Assume text is inside <p> tags

    policy_text = "\n".join([para.get_text() for para in paragraphs])
    return policy_text


def save_policies_to_file(urls, output_file):
    """
    Fetches policies from a list of URLs and saves them to a file.
    """
    policies = []

    for url in urls:
        print(f"Fetching policy from {url}...")
        policy_text = fetch_policy(url)
        policies.append(policy_text)

    # Save policies to a CSV
    import pandas as pd
    df = pd.DataFrame(policies, columns=['text'])
    df.to_csv(output_file, index=False)
    print(f"Saved policies to {output_file}")


urls = [
    "https://policies.google.com/privacy",
    "https://www.facebook.com/policy.php",
    "https://www.shopify.com/legal/privacy",
    "https://storage.googleapis.com/etsy-extfiles-prod/2024%20Privacy%20Policy/Privacy%20Policy_en_US.docx.pdf",
    "https://www.netflix.com/privacy",
    "https://slack.com/privacy-policy",
    "https://zoom.us/privacy",
    "https://www.dropbox.com/privacy",
    "https://www.linkedin.com/legal/privacy-policy",
]


save_policies_to_file(urls, "data/privacy_policies.csv")
