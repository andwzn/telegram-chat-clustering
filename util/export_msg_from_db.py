
import os
import pandas as pd
import glob
from data.dbs.schema import Chat, Message, MessageVersion, ChatVersion, UserVersion, User, Webpage
from sqlalchemy.orm import aliased, Session as DBSession, Session, sessionmaker
from sqlalchemy import Subquery, and_, func, Row, case, create_engine
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Tuple
from sqlalchemy.engine.row import Row


def main() -> None:
    """
    Export messages from a SQLite database to a CSV file.

    This function connects to a SQLite database, retrieves all messages from the database,
    and exports them as a CSV file.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(base_dir, '../data/dbs/')
        csv_dir = os.path.join(base_dir, '../data/csv/')
            
        # Get list of all .db files in the db directory
        db_files = glob.glob(os.path.join(db_dir, '*.db'))         
                
        
        # Loop through each db file
        for db_file in db_files:
            db_name = os.path.basename(db_file)
            csv_file = os.path.join(csv_dir, f"{os.path.splitext(db_name)[0]}.csv")

            # Connect to db
            engine = create_engine(f"sqlite:///{db_file}", echo=True)
            Session = sessionmaker(bind=engine)
            with Session() as session:
                # Get all messages
                msgs = getMessagesFromDB(session)

            # Export messages as csv‚
            msgs_csv = dictsToCsv(rowsToDicts(msgs))
            with open(csv_file, 'w') as f:
                f.write(msgs_csv)

            # Close the session
            session.close()
            
        
    except:
        print("Error converting messages to csv")
        raise
    
    

def getLatestVersionSubqueries(session: DBSession) -> Tuple[Subquery, Subquery, Subquery]:
    """
    Generate subqueries to find the latest version of chats, messages, and users.

    Parameters:
    session (Session): The SQLAlchemy session to use for the queries.

    Returns:
    Tuple[Subquery, Subquery, Subquery]: A tuple containing three subqueries:
        - The first subquery returns the latest version of chats.
        - The second subquery returns the latest version of messages.
        - The third subquery returns the latest version of users.
    """
    try:
        base_query = (
            session.query(
                Chat.id.label('chat_id'),
                Chat.external_id.label('chat_external_id'),
                Message.id.label('message_id'),
                Message.date.label('message_date'),
                User.id.label('user_id'),
                UserVersion.id.label('user_version_id'),
                ChatVersion.id.label('chat_version_id'),
                MessageVersion.id.label('message_version_id')
            )
            .join(Message, Chat.id == Message.chat_id)
            .join(MessageVersion, Message.id == MessageVersion.message_id)
            .join(User, User.id == Message.sender_id)
            .join(UserVersion, User.id == UserVersion.user_id)
            .join(ChatVersion, Chat.id == ChatVersion.chat_id)
        ).subquery()

        latest_chat_version_subquery = (
            session.query(
                base_query.c.chat_id,
                func.max(ChatVersion.created_at).label('max_create_date')
            )
            .join(ChatVersion, base_query.c.chat_version_id == ChatVersion.id)
            .group_by(base_query.c.chat_id)
            .subquery('latest_chat_version_subquery')
        )

        latest_message_version_subquery = (
            session.query(
                base_query.c.message_id,
                func.max(MessageVersion.created_at).label('max_create_date')
            )
            .join(MessageVersion, base_query.c.message_version_id == MessageVersion.id)
            .group_by(base_query.c.message_id)
            .subquery('latest_message_version_subquery')
        )

        latest_user_version_subquery = (
            session.query(
                base_query.c.user_id,
                func.max(UserVersion.created_at).label('max_create_date')
            )
            .join(UserVersion, base_query.c.user_version_id == UserVersion.id)
            .group_by(base_query.c.user_id)
            .subquery('latest_user_version_subquery')
        )

        return latest_chat_version_subquery, latest_message_version_subquery, latest_user_version_subquery
    
    except Exception as e:
        print("Error in get_latest_version_subqueries from db: ", e)
        raise e
    
def getMessagesFromDB(
        session: DBSession
)-> List[Row]:
    """
    Retrieves messages from the database.

    Args:
        session (DBSession): The database session object.

    Returns: List[Row]: A list of messages.
        
    """    
    try:

        # get subqueries to filter for latest iteration of versionized data
        latest_chat_version_subquery, latest_message_version_subquery, latest_user_version_subquery = getLatestVersionSubqueries(session)

        # Alias chat table to get source chat of forwarded messages
        source_chat = aliased(Chat)

        # Alias User Version table to get source users of forwarded messages
        source_user = aliased(UserVersion)

        messages = (
            session.query(
                Message.external_id.label('telegram_message_id'),
                MessageVersion.normalized_text.label('message_text'),
                Message.date.label("message_date"),
                MessageVersion.reactions.label('message_reactions'),
                MessageVersion.reactions_count.label("message_reactions_count"),
                MessageVersion.forwards_count.label('message_fwd_count'),
                MessageVersion.views_count.label('message_view_count'),
                Message.media_type.label("message_media_type"),
                case(
                    (Message.grouped_id != None, True),
                    else_=False
                ).label('is_group_elem'),
                Message.grouped_id.label("message_group_id"),
                User.external_id.label("telegram_sender_id"),
                UserVersion.first_name.label("sender_first_name"),
                UserVersion.last_name.label("sender_last_name"),
                UserVersion.username.label("sender_username"),
                UserVersion.display_name.label("sender_display_name"),
                Message.post_author.label("post_author"),
                Chat.external_id.label("telegram_chat_id"),
                ChatVersion.chat_name.label("chat_name"),
                Chat.username.label("chat_handle"),
                case(
                    (Message.fwd_date != None, True),
                    else_=False
                ).label('is_fwd'),
                source_chat.username.label("fwd_from_chat_handle"),
                source_user.first_name.label("fwd_from_user_name"),
                case(
                        (Message.reply_to_id != None, True),
                        (Message.reply_to_top_id != None, True),
                    else_=False
                ).label('is_reply'),
                Message.reply_to_id.label("reply_to_message_id"),
                Message.reply_to_top_id.label("reply_to_top_message_id"),
                Message.created_at.label("collection_time"), 
                Webpage.author.label("webpage_author"), 
                Webpage.title.label("webpage_title"), 
                Webpage.description.label("webpage_description"),
                Webpage.url.label("webpage_url")
            )
            .join(MessageVersion, Message.id == MessageVersion.message_id)
            .join(Chat, Message.chat_id == Chat.id)
            .join(ChatVersion, Chat.id == ChatVersion.chat_id)
            .join(User, User.id == Message.sender_id)
            .join(UserVersion, User.id == UserVersion.user_id)
            .join(latest_message_version_subquery, and_(
                MessageVersion.message_id == latest_message_version_subquery.c.message_id,
                MessageVersion.created_at == latest_message_version_subquery.c.max_create_date,
            ))
            .join(latest_user_version_subquery, and_(
                UserVersion.user_id == latest_user_version_subquery.c.user_id,
                UserVersion.created_at == latest_user_version_subquery.c.max_create_date
            ))
            .join(latest_chat_version_subquery, and_(
                ChatVersion.chat_id == latest_chat_version_subquery.c.chat_id,
                ChatVersion.created_at == latest_chat_version_subquery.c.max_create_date
            ))
            .outerjoin(source_chat, source_chat.id == Message.fwd_from_chat_id)
            .outerjoin(source_user, source_user.id == Message.fwd_from_user_id)
            .outerjoin(Webpage, MessageVersion.webpage_id == Webpage.id)
            .order_by(Message.date.desc())
            .all()
        )

        return messages
    
    except Exception as e:
        print("Error getting messages from DB: ", e)
        raise e

class MessageDict(TypedDict):
    """
    Represents a dictionary structure for storing information about a message.

    Attributes:
        telegram_message_id (int): The ID of the Telegram message.
        message_text (str): The text content of the message.
        message_date (datetime): The date and time when the message was sent.
        message_reactions (Optional[dict[str, int]]): A dictionary of message reactions and their counts.
        message_reactions_count (Optional[int]): The total count of message reactions.
        message_fwd_count (Optional[int]): The count of times the message has been forwarded.
        message_view_count (Optional[int]): The count of times the message has been viewed.
        message_media_type (Optional[str]): The type of media in the message (e.g., photo, video).
        is_group_elem (bool): Indicates whether the message is part of a album. 
        message_group_id (Optional[int]): The ID of the album the message belongs to.
        telegram_sender_id (Optional[int]): The ID of the sender of the message.
        sender_first_name (Optional[str]): The first name of the sender.
        sender_last_name (Optional[str]): The last name of the sender.
        sender_username (Optional[str]): The username of the sender.
        sender_display_name (Optional[str]): The display name of the sender.
        post_author (Optional[str]): The author of the post (if applicable).
        telegram_chat_id (int): The ID of the Telegram chat the message was sent in.
        chat_name (Optional[str]): The name of the chat.
        chat_handle (Optional[str]): The handle of the chat.
        is_fwd (bool): Indicates whether the message is a forwarded message.
        fwd_from_chat_handle (Optional[str]): The handle of the chat from which the message was forwarded.
        fwd_from_user_name (Optional[str]): The username of the user from which the message was forwarded.
        is_reply (bool): Indicates whether the message is a reply to another message.
        reply_to_message_id (Optional[int]): The ID of the message being replied to.
        reply_to_top_message_id (Optional[int]): The ID of the top-level message of the thread being replied to.
        collection_time (datetime): The date and time when the message was collected.
    """
    telegram_message_id: int
    message_text: str
    message_date: datetime
    message_reactions: Optional[dict[str, int]]
    message_reactions_count: Optional[int]
    message_fwd_count: Optional[int]
    message_view_count: Optional[int]
    message_media_type: Optional[str]
    is_group_elem: bool
    message_group_id: Optional[int]
    telegram_sender_id: Optional[int]
    sender_first_name: Optional[str]
    sender_last_name: Optional[str]
    sender_username: Optional[str]
    sender_display_name: Optional[str]
    post_author: Optional[str]
    telegram_chat_id: int
    chat_name: Optional[str]
    chat_handle: Optional[str]
    is_fwd: bool
    fwd_from_chat_handle: Optional[str]
    fwd_from_user_name: Optional[str]
    is_reply: bool
    reply_to_message_id: Optional[int]
    reply_to_top_message_id: Optional[int]
    collection_time: datetime
    
    
def rowsToDicts(rows: List[Row]) -> List[MessageDict]:
    """
    Converts a list of SQAlchemy Row objects to a list of dictionaries.

    Args:
        rows (List[Row]): A list of Row objects.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary represents a row.

    """
    try:
        return [row._mapping for row in rows]
    
    except Exception as e:
        print("Error converting sqlalchemy rows to dicts: ", e)
        raise e
    

def dictsToCsv(dicts: List[Dict]) -> str:
    """
    Convert a list of dictionaries to a CSV string.

    Args:
        dicts (Dict): The dictionaries to be converted.

    Returns:
        str: The CSV string representation of the dictionaries.
    """
    
    try:
        if not dicts:
            return ''

        df = pd.DataFrame(dicts)
        csv_string = df.to_csv(index=True, na_rep='NaN')
        return csv_string

    except Exception as e:
        print("Error converting dict to csv: ", e)
        raise e    
    
if __name__ == "__main__":
    main()
        