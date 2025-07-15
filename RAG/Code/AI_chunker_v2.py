# Adapted from https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/a4570f3c4883eb9b835b0ee18990e62298f518ef/tutorials/LevelsOfTextSplitting/agentic_chunker.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI
import uuid
import os
from typing import Optional
from pydantic import BaseModel
from langchain.chains import create_extraction_chain_pydantic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AgenticChunker:
    def __init__(self, openai_api_key=None):
        self.chunks = {}
        self.id_truncate_limit = 5

        # Whether or not to update/refine summaries and titles as you get new information
        self.generate_new_metadata_ind = True
        self.print_logging = True

        if openai_api_key is None:
            openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if openai_api_key is None:
            raise ValueError(
                "API key is not provided and not found in environment variables")

        self.llm = AzureChatOpenAI(
            openai_api_key=openai_api_key,
            openai_api_version="YOUR API_version",  # Your API_version
            azure_endpoint="YOUR_URL",  # Your endpoint
            azure_deployment="YOUR_MODEL",  # Your model_deployment
            model="YOUR_MODEL_NAME",  # Your model_name
            validate_base_url=False,
        )

    def add_propositions(self, propositions):
        """Add multiple propositions to the chunker"""
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition):
        """Add a single proposition to the appropriate chunk"""
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        # If a chunk was found then add the proposition to it
        if chunk_id:
            if self.print_logging:
                print(
                    f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition)
            return
        else:
            if self.print_logging:
                print("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition)

    def add_proposition_to_chunk(self, chunk_id, proposition):
        """Add proposition to existing chunk and update metadata"""
        # Add the proposition
        self.chunks[chunk_id]['propositions'].append(proposition)

        # Then grab a new summary
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(
                self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(
                self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        """Update chunk summary when new proposition is added"""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunks current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk new summary, nothing else.
                    """,
                ),
                ("user",
                 "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary']
        }).content

        return new_chunk_summary

    def _update_chunk_title(self, chunk):
        """Update chunk title when new proposition is added"""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user",
                 "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )

        runnable = PROMPT | self.llm

        updated_chunk_title = runnable.invoke({
            "proposition": "\n".join(chunk['propositions']),
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        }).content

        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        """Generate summary for new chunk"""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user",
                 "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": proposition
        }).content

        return new_chunk_summary

    def _get_new_chunk_title(self, summary):
        """Generate title for new chunk"""
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about

                    You will be given a summary of a chunk which needs a title

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user",
                 "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_title = runnable.invoke({
            "summary": summary
        }).content

        return new_chunk_title

    def _create_new_chunk(self, proposition):
        """Create a new chunk with the given proposition"""
        # Generate short ID
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [proposition],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }

        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def get_chunk_outline(self):
        """Get a string representation of all chunks"""
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"""
            chunk_outline += single_chunk_string

        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        """Find the most relevant existing chunk for a proposition"""
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine whether or not the "Proposition" should belong to any of the existing chunks.

                    A proposition should belong to a chunk if their meaning, direction, or intention are similar.
                    The goal is to group similar propositions and chunks.

                    If you think a proposition should be joined with a chunk, return the chunk id.
                    If you do not think an item should be joined with an existing chunk, just return "No chunks"

                    Example:
                    Input:
                        - Proposition: "Greg really likes hamburgers"
                        - Current Chunks:
                            - Chunk ID: 2n4l3d
                            - Chunk Name: Places in San Francisco
                            - Chunk Summary: Overview of the things to do with San Francisco Places

                            - Chunk ID: 93833k
                            - Chunk Name: Food Greg likes
                            - Chunk Summary: Lists of the food and dishes that Greg likes
                    Output: 93833k
                    """,
                ),
                ("user",
                 "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user",
                 "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        chunk_found = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).content

        # Pydantic data class (compatible with Pydantic v2)
        class ChunkID(BaseModel):
            """Extracting the chunk id"""
            chunk_id: Optional[str] = None

        # Extraction to catch-all LLM responses with error handling
        try:
            extraction_chain = create_extraction_chain_pydantic(
                pydantic_schema=ChunkID, llm=self.llm)
            extraction_found = extraction_chain.run(chunk_found)
            if extraction_found and len(extraction_found) > 0:
                chunk_found = extraction_found[0].chunk_id
        except Exception as e:
            if self.print_logging:
                print(f"Warning: Extraction chain failed: {e}")
            # Fallback: try to extract chunk ID directly from response
            lines = chunk_found.strip().split('\n')
            for line in lines:
                line = line.strip()
                if len(line) == self.id_truncate_limit and line.replace('-', '').replace('_', '').isalnum():
                    chunk_found = line
                    break

        # Validate chunk ID format
        if not chunk_found or len(chunk_found) != self.id_truncate_limit:
            return None

        # Verify chunk exists
        if chunk_found not in self.chunks:
            return None

        return chunk_found

    def get_chunks(self, get_type='dict'):
        """
        Return chunks in specified format

        Args:
            get_type (str): 'dict' returns full chunk data, 'list_of_strings' returns concatenated propositions

        Returns:
            dict or list: Chunks in requested format
        """
        if get_type == 'dict':
            return self.chunks
        elif get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk['propositions']]))
            return chunks
        else:
            raise ValueError("get_type must be 'dict' or 'list_of_strings'")

    def get_chunks_for_embedding(self):
        """
        Get chunks formatted specifically for embedding

        Returns:
            list: List of dictionaries with chunk metadata and content
        """
        embedding_chunks = []
        for chunk_id, chunk in self.chunks.items():
            embedding_chunks.append({
                'chunk_id': chunk_id,
                'title': chunk['title'],
                'summary': chunk['summary'],
                'content': " ".join(chunk['propositions']),
                'proposition_count': len(chunk['propositions']),
                'metadata': {
                    'chunk_index': chunk['chunk_index'],
                    'propositions': chunk['propositions']
                }
            })
        return embedding_chunks

    def pretty_print_chunks(self):
        """Print a formatted view of all chunks"""
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Title: {chunk['title']}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions ({len(chunk['propositions'])}):")
            for i, prop in enumerate(chunk['propositions'], 1):
                print(f"    {i}. {prop}")
            print("\n" + "-"*50 + "\n")

    def pretty_print_chunk_outline(self):
        """Print a brief outline of all chunks"""
        print("Chunk Outline\n")
        print(self.get_chunk_outline())

    def get_stats(self):
        """Get statistics about the chunks"""
        total_propositions = sum(len(chunk['propositions'])
                                 for chunk in self.chunks.values())

        stats = {
            'total_chunks': len(self.chunks),
            'total_propositions': total_propositions,
            'avg_propositions_per_chunk': total_propositions / len(self.chunks) if self.chunks else 0,
            'chunk_sizes': [len(chunk['propositions']) for chunk in self.chunks.values()],
            'largest_chunk_size': max([len(chunk['propositions']) for chunk in self.chunks.values()]) if self.chunks else 0,
            'smallest_chunk_size': min([len(chunk['propositions']) for chunk in self.chunks.values()]) if self.chunks else 0
        }

        return stats

    def print_stats(self):
        """Print chunk statistics"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("CHUNK STATISTICS")
        print("="*50)
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Total Propositions: {stats['total_propositions']}")
        print(
            f"Average Propositions per Chunk: {stats['avg_propositions_per_chunk']:.2f}")
        print(f"Largest Chunk: {stats['largest_chunk_size']} propositions")
        print(f"Smallest Chunk: {stats['smallest_chunk_size']} propositions")
        print("="*50)

    def reset(self):
        """Reset the chunker to empty state"""
        self.chunks = {}
        if self.print_logging:
            print("Chunker reset - all chunks cleared")

    def set_logging(self, enable_logging):
        """Enable or disable logging"""
        self.print_logging = enable_logging


# Convenience function for easy import
def create_agentic_chunker(api_key=None, enable_logging=True):
    """
    Create and return an AgenticChunker instance

    Args:
        api_key (str, optional): Azure OpenAI API key
        enable_logging (bool): Whether to enable logging output

    Returns:
        AgenticChunker: Configured chunker instance
    """
    chunker = AgenticChunker(openai_api_key=api_key)
    chunker.set_logging(enable_logging)
    return chunker


if __name__ == "__main__":
    # Test the chunker with sample propositions
    print("Testing AgenticChunker...")

    sample_propositions = [
        'The month is October.',
        'The year is 2023.',
        "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
        'Teachers and coaches implicitly told us that the returns were linear.',
        "I heard a thousand times that 'You get out what you put in.'",
        'The same pattern of superlinear returns is observed in fame.',
        'The same pattern of superlinear returns is observed in power.',
    ]

    # Create chunker
    ac = create_agentic_chunker()

    # Add propositions
    ac.add_propositions(sample_propositions)

    # Display results
    ac.pretty_print_chunks()
    ac.print_stats()

    print("\n✅ AgenticChunker test completed successfully!")
