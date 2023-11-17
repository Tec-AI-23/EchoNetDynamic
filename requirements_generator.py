import requests


def get_latest_version_of_package(package_name):
    # Fetches directly from the pip website the value of the default/latest version

    url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"'{package_name}' not found on pip")
        return None

    data = response.json()
    latest_version = data.get("info", {}).get("version")
    print(f"{package_name}=={latest_version}")

    return latest_version


if __name__ == "__main__":
    packages_names_file = "PACKAGES_NAMES.txt"
    latest_versions_dir = {}
    with open(packages_names_file, "r") as file:
        for line in file.readlines():
            package_name = line[:-1]
            latest_version = get_latest_version_of_package(package_name)
            if latest_version:
                latest_versions_dir[package_name] = latest_version

    with open("requirements.txt", "w+") as file:
        for package in latest_versions_dir:
            package_string = f"{package}=={latest_versions_dir[package]}\n"
            file.write(package_string)
