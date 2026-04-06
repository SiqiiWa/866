#!/usr/bin/env python3
"""
Simple script to view saved episodes from the database.
Shows how to retrieve and display dialogue results with a specific tag.
"""

import asyncio
import json
from sotopia.database import EpisodeLog


def view_episodes_by_tag(tag: str = "qwen_test"):
    """
    Retrieve and display all episodes with a given tag.
    
    Args:
        tag: The tag used when running episodes (default: "qwen_test")
    """
    try:
        # Query episodes by tag
        episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
        
        if not episodes:
            print(f"No episodes found with tag: {tag}")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(episodes)} episodes with tag: {tag}")
        print(f"{'='*60}\n")
        
        for idx, episode in enumerate(episodes, 1):
            print(f"Episode {idx}:")
            print(f"  ID: {episode.pk}")
            print(f"  Environment: {episode.environment}")
            print(f"  Agents: {', '.join(episode.agents)}")
            print(f"  Models: {', '.join(episode.models if episode.models else [])}")
            print(f"  Tag: {episode.tag}")
            print(f"  Number of turns: {len(episode.messages)}")
            print(f"  Reasoning: {episode.reasoning[:100]}..." if len(episode.reasoning) > 100 else f"  Reasoning: {episode.reasoning}")
            print(f"  Rewards: {episode.rewards}")
            print()
    
    except Exception as e:
        print(f"Error retrieving episodes: {e}")
        print("\nNote: Make sure the database is running and configured correctly.")
        print("If using local storage, episodes should be in the database directory.")


def view_episode_messages(episode_id: str):
    """
    Display detailed messages for a specific episode.
    
    Args:
        episode_id: The primary key of the episode
    """
    try:
        episode = EpisodeLog.get(episode_id)
        
        if not episode:
            print(f"Episode not found: {episode_id}")
            return
        
        print(f"\n{'='*60}")
        print(f"Episode Detail: {episode_id}")
        print(f"{'='*60}\n")
        
        for turn_idx, turn_messages in enumerate(episode.messages, 1):
            print(f"Turn {turn_idx}:")
            for sender, recipient, message in turn_messages:
                print(f"  {sender} -> {recipient}: {message}")
            print()
        
        print(f"Reasoning: {episode.reasoning}")
        print(f"Rewards: {episode.rewards}")
    
    except Exception as e:
        print(f"Error retrieving episode: {e}")


def export_episodes_to_json(tag: str = "qwen_test", output_file: str = "episodes.json"):
    """
    Export episodes with a given tag to a JSON file.
    
    Args:
        tag: The tag used when running episodes
        output_file: Output JSON file path
    """
    try:
        episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
        
        if not episodes:
            print(f"No episodes found with tag: {tag}")
            return
        
        # Convert episodes to JSON-serializable format
        episodes_data = []
        for episode in episodes:
            episodes_data.append({
                "id": episode.pk,
                "environment": episode.environment,
                "agents": episode.agents,
                "tag": episode.tag,
                "models": episode.models,
                "messages": [
                    [(m[0], m[1], m[2]) for m in turn]
                    for turn in episode.messages
                ],
                "reasoning": episode.reasoning,
                "rewards": episode.rewards,
            })
        
        with open(output_file, 'w') as f:
            json.dump(episodes_data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(episodes_data)} episodes to {output_file}")
    
    except Exception as e:
        print(f"Error exporting episodes: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list" and len(sys.argv) > 2:
            view_episodes_by_tag(sys.argv[2])
        elif command == "show" and len(sys.argv) > 2:
            view_episode_messages(sys.argv[2])
        elif command == "export" and len(sys.argv) > 2:
            tag = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else f"episodes_{tag}.json"
            export_episodes_to_json(tag, output_file)
        else:
            print("Usage:")
            print("  python view_episodes.py list [tag]           - List episodes by tag")
            print("  python view_episodes.py show <episode_id>    - Show episode details")
            print("  python view_episodes.py export <tag> [file]  - Export episodes to JSON")
            print("\nDefault tag is 'qwen_test'")
    else:
        # Default: show episodes with default tag
        view_episodes_by_tag()
