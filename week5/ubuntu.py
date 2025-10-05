#!/usr/bin/env python3
"""
Ubuntu Image Fetcher
A tool that embodies Ubuntu principles: community, respect, sharing, and practicality
"""

import os
import requests
from urllib.parse import urlparse, unquote
from pathlib import Path


def create_directory(dir_name="Fetched_Images"):
    """Create the directory for storing images if it doesn't exist."""
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def extract_filename(url):
    """Extract filename from URL or generate one."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    filename = os.path.basename(path)
    
    # If no filename found or it doesn't have an extension, generate one
    if not filename or '.' not in filename:
        filename = f"image_{hash(url) % 10000}.jpg"
    
    return filename


def download_image(url, save_dir):
    """
    Download an image from the provided URL.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"\n🌍 Connecting to the global community...")
        print(f"📡 Fetching: {url}")
        
        # Send request with a timeout
        response = requests.get(url, timeout=10, stream=True)
        
        # Respect the server's response - check for errors
        response.raise_for_status()
        
        # Verify it's actually an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            print(f"⚠️  Warning: The resource might not be an image (Content-Type: {content_type})")
            print("   Proceeding anyway...")
        
        # Extract filename and create full path
        filename = extract_filename(url)
        filepath = os.path.join(save_dir, filename)
        
        # Save the image in binary mode
        print(f"💾 Saving to: {filepath}")
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(filepath)
        print(f"✅ Success! Downloaded {file_size:,} bytes")
        print(f"📁 Image saved in community folder: {save_dir}/")
        
        return True
        
    except requests.exceptions.Timeout:
        print("⏱️  Connection timeout - the server took too long to respond.")
        print("   Please try again or check your internet connection.")
        return False
        
    except requests.exceptions.HTTPError as e:
        print(f"🚫 HTTP Error {e.response.status_code}: {e.response.reason}")
        print("   The server couldn't fulfill the request.")
        return False
        
    except requests.exceptions.ConnectionError:
        print("🔌 Connection error - couldn't reach the server.")
        print("   Please check the URL and your internet connection.")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
        
    except IOError as e:
        print(f"💿 File system error: {e}")
        print("   Couldn't save the file to disk.")
        return False
        
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")
        return False


def main():
    """Main function implementing Ubuntu principles."""
    print("=" * 60)
    print("Welcome to the Ubuntu Image Fetcher")
    print("🤝 Ubuntu Image Fetcher")
    print("   'I am because we are' - Connecting to shared resources")
    print("=" * 60)
    
    # Create the community directory
    save_dir = create_directory()
    print(f"\n📂 Community directory ready: {save_dir}/")
    
    # Prompt the user for a URL
    url = input("\n🔗 Please enter the image URL: ").strip()
    
    if not url:
        print("❌ No URL provided. Exiting gracefully.")
        return
    
    # Validate URL format
    if not url.startswith(('http://', 'https://')):
        print("⚠️  URL should start with http:// or https://")
        print("   Adding https:// prefix...")
        url = 'https://' + url
    
    # Download the image
    success = download_image(url, save_dir)
    
    if success:
        print("\n🎉 Mission accomplished with Ubuntu spirit!")
        print("   Your image is ready for sharing and appreciation.")
    else:
        print("\n🙏 Thank you for trying. Not all connections succeed,")
        print("   but the community spirit remains strong.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()