from sqlalchemy import (
    Boolean,
    func,
    Identity,
    ForeignKey,
    DateTime,
    String,
    Integer,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship
from sqlalchemy.ext.declarative import declared_attr
from datetime import datetime
from typing import List


class Base(DeclarativeBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(), onupdate=func.now()
    )


class ChatVersion(Base):
    __tablename__ = "chat_versions"

    chat: Mapped["Chat"] = relationship(back_populates="versions")
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"))

    """
    Chat snapshots referring to this version. 
    """
    snapshots: Mapped[List["ChatSnapshot"]] = relationship(back_populates="chat_version")

    chat_name: Mapped[str] = mapped_column(String, nullable=True)

    chat_description: Mapped[str] = mapped_column(String, nullable=True)

    member_count: Mapped[int] = mapped_column(Integer, nullable=True)

    chat_type: Mapped[str] = mapped_column(String, nullable=True) # values: broadcast, group, megagroup, unknown

    pinned_msg: Mapped["Message"] = relationship(back_populates="pinned_in_versions")
    
    pinned_msg_id: Mapped[int] = mapped_column(ForeignKey('messages.id'), nullable=True)
    
    is_private: Mapped[bool] = mapped_column(Boolean, nullable=True, default=False) 

class ChatSnapshot(Base):
    """
    Contains Snapshots of chats that were scraped. 
    If TeleVision finds messages forwarded from another chat, it also collects the "original" version of said message
    from the source chat.
    The source chat itself is included in the chat and chat version tables, but not in the snapshots table,
    as it is not scraped in a specific timeframe but only to obtain these specific messages.
    """

    __tablename__ = "chat_snapshots"

    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"))
    chat: Mapped["Chat"] = relationship(back_populates="snapshots")

    """
    The chat version this snapshot refers to. This means, the latest version of the chat at the time of the snapshot.
    """
    chat_version_id: Mapped[int] = mapped_column(ForeignKey("chat_versions.id"))
    chat_version: Mapped["ChatVersion"] = relationship(back_populates="snapshots")

    scrape_timeframe_from: Mapped[datetime] = mapped_column(DateTime)

    scrape_timeframe_to: Mapped[datetime] = mapped_column(DateTime)

    """
    Whether the snapshot was creted as the enrtry point for a scaping process, i.e. directly requested by user and not scraped as part of a network
    """
    is_entry_point: Mapped[bool] = mapped_column(Boolean, default=False) # TODO: Move to separate table "scraping_processes" or similar

    """
    Only set if is_entry_point is True. The original configuration of the scraping process.
    """
    scraping_config: Mapped[dict] = mapped_column(type_=JSON, nullable=True) # TODO: Move to separate table "scraping_processes" or similar
    
    """
    Was the chat scraped successfully in this timeframe? It's only set to False, when the scraping process was interrupted or failed in a unforseen way.
    It is also possible that the chat was not scraped due to it being private or a user. In this case, the flag is set to True, as TeleVision can handle such cases.
    """
    scraped_successfully: Mapped[bool] = mapped_column(Boolean, default=False) # TODO: Move to separate table "scraping_processes" or similar


class User(Base):
    __tablename__ = "users"
    
    """
    The user ID from the Telegram API
    """
    external_id: Mapped[int] = mapped_column(Integer)
    
    """
    is this user a bot?
    """
    is_bot: Mapped[bool] = mapped_column(Boolean, nullable=True)    
    
    """
    messages sent by the user
    """
    messages: Mapped[List["Message"]] = relationship(back_populates="sender", foreign_keys="Message.sender_id")
    
    
    """
    Versions of the user
    """
    versions: Mapped[List["UserVersion"]] = relationship(back_populates="user")    
    
    """
    Forwarded messages (as appearing in other chats) originating from this user 
    """
    forwarded_messages: Mapped[List["Message"]] = relationship(
        back_populates="fwd_from_user", 
        foreign_keys="Message.fwd_from_user_id"
    )

class UserVersion(Base):
    
    __tablename__ = "user_versions"
    
    
    """
    username of the user
    """
    username: Mapped[str] = mapped_column(String, nullable=True)
    
    """
    usernames of the user
    """
    additional_usernames: Mapped[List[str]] = mapped_column(JSON, nullable=True)    
    
    """
    first name of the user
    """
    first_name: Mapped[str] = mapped_column(String, nullable=True)
    
    """
    last name of the user
    """
    last_name: Mapped[str] = mapped_column(String, nullable=True)
    
    """
    display name of the user
    """
    display_name: Mapped[str] = mapped_column(String, nullable=True)
    
    """
    was this user reported as a fake user?
    """
    is_fake: Mapped[bool] = mapped_column(Boolean, nullable=True)
    
    """
    user may be spam user
    """
    is_scam: Mapped[bool] = mapped_column(Boolean, nullable=True)
    
    """
    user may be a member of a restricted group
    """
    is_restricted: Mapped[bool] = mapped_column(Boolean, nullable=True)
    
    """
    user is veryfied
    """
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=True)    
    
    """
    phone number of the user
    """
    phone: Mapped[str] = mapped_column(String, nullable=True)    
    
    """
    The user this version belongs to
    """
    user: Mapped["User"] = relationship(back_populates="versions")
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))    
    

class Chat(Base):
    __tablename__ = "chats"

    """
    The chat ID from the Telegram API
    """
    external_id: Mapped[int] = mapped_column(Integer)

    """
    The date of the first message in the chat
    """
    initial_msg_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    """
    The messages in the chat
    """
    messages: Mapped[List["Message"]] = relationship(back_populates="chat", foreign_keys="Message.chat_id")

    """
    Forwarded messages (as appearing other chats) originating from this chat 
    """
    forwarded_messages: Mapped[List["Message"]] = relationship(
        back_populates="fwd_from_chat", 
        foreign_keys="Message.fwd_from_chat_id"
    )

    """
    The snapshots of the chat, i.e. scrape processes
    """
    snapshots: Mapped[List["ChatSnapshot"]] = relationship(back_populates="chat")
    
    """
    The versions of the chat
    """
    versions: Mapped[List["ChatVersion"]] = relationship(back_populates="chat")

    """
    The username assiciated with the chat
    """
    username: Mapped[str] = mapped_column(String, nullable=True)

    """
    Whether the chat is currently being scraped
    """
    is_scraping: Mapped[bool] = mapped_column(Boolean, default=False) # TODO: Move to separate table "chat_status" or similar

    """
    A description given to the chat by the user
    """
    user_description: Mapped[str] = mapped_column(String, nullable=True) # TODO: Move to separate table "chat_status" or similar
    
    """
    Actions performed in/on this chat
    """
    actions: Mapped[List["Actions"]] = relationship("Actions", back_populates="chat")


class Message(Base):
    __tablename__ = "messages"

    """
    The message ID from the Telegram API
    """
    external_id: Mapped[int] = mapped_column(Integer)

    """
    The chat the message was sent in
    """
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"))
    chat: Mapped["Chat"] = relationship(back_populates="messages", foreign_keys=[chat_id])

    """
    The date the message was sent
    """
    date: Mapped[datetime] = mapped_column(DateTime)

    """
    Data about the sender of the message. JSON format.
    """
    #sender_data: Mapped[dict] = mapped_column(type_=JSON) 

    """
    The ID of the sender of the message from the Telegram API
    """
    #sender_external_id: Mapped[int] = mapped_column(Integer)
    
    """
    The ID of the sender
    """
    sender_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    sender: Mapped["User"] = relationship(back_populates="messages", foreign_keys=[sender_id])
    
    """
    Post author (only set, if message has no sender)
    """
    post_author: Mapped[str] = mapped_column(String, nullable=True)

    """
    ID of the inline bot that generated the message
    """
    via_bot_id: Mapped[int] = mapped_column(Integer, nullable=True)

    """
    The date the message was forwarded
    """
    fwd_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    
    """
    Id of the message the message is grouped with (Same grouped-ID indicates multimedia-messages sent as an album or media group)
)
    """
    grouped_id: Mapped[int] = mapped_column(Integer, nullable=True)

    """
    The name of the entity a messages was forwarded from (not always available, maybe limited to messges from users?)
    """
    fwd_name: Mapped[str] = mapped_column(String, nullable=True)

    """
    The chat the message was forwarded from
    """
    fwd_from_chat_external_id: Mapped[int] = mapped_column(Integer, nullable=True)
    fwd_from_chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"), nullable=True)
    fwd_from_chat: Mapped["Chat"] = relationship(back_populates="forwarded_messages", foreign_keys=[fwd_from_chat_id]) 
    
    """
    The user the message was forwarded from
    """
    fwd_from_user_external_id: Mapped[int] = mapped_column(Integer, nullable=True)
    fwd_from_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    fwd_from_user: Mapped["User"] = relationship(back_populates="forwarded_messages", foreign_keys=[fwd_from_user_id])     

    """
    The forwarded message as it appeared in its origin chat
    """
    original_message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=True)
    original_message: Mapped["Message"] = relationship(back_populates="fwd_instances", remote_side="Message.id", foreign_keys=[original_message_id]) 
    
    """
    Instances of a messages as they appear in other chats
    """
    fwd_instances: Mapped[List["Message"]]  = relationship(back_populates="original_message", foreign_keys=[original_message_id])

    """
    The ID of the message the message was a reply to
    """
    reply_to_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=True)
    reply_to: Mapped["Message"] = relationship(back_populates="replies", foreign_keys=[reply_to_id])
    
    """
    Direct replies to this messages
    """
    replies: Mapped[List["Message"]] = relationship(back_populates="reply_to", foreign_keys=[reply_to_id], remote_side="Message.id")    
    
    """
    The topmost message in a thread the message is a reply to / message that started a thread
    """
    reply_to_top_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=True)
    reply_to_top: Mapped["Message"] = relationship(back_populates="replies_in_thread", foreign_keys=[reply_to_top_id])
    
    """
    Indirect (and direct?) replies to this message
    """
    replies_in_thread: Mapped[List["Message"]] = relationship(back_populates="reply_to_top", foreign_keys=[reply_to_top_id], remote_side="Message.id")
    

    """
    Versions of the message
    """
    versions: Mapped[List["MessageVersion"]] = relationship(back_populates="message")

    # media = Column(JSON)

    """
    Chat versions the message the message was pinned in
    """
    pinned_in_versions: Mapped[List["ChatVersion"]] = relationship(back_populates="pinned_msg")
    
    """
    Type of media attachted to the message. 
    """
    media_type: Mapped[str] = mapped_column(String, nullable=True)
    
    """
    Url of sent webpage(s)
    """
    referenced_url: Mapped[str] = mapped_column(String, nullable=True)


class MessageVersion(Base):
    __tablename__ = "message_versions"

    """
    Text of the message
    """
    text: Mapped[str] = mapped_column(String, nullable=True)

    """
    Normalized message text. Normalize emoji, line breaks etc.
    """
    normalized_text: Mapped[str] = mapped_column(String, nullable=True)

    """
    Reaction data. JSON format.
    """
    reactions: Mapped[dict] = mapped_column(type_=JSON, nullable=True)

    """
    Total number of reactions
    """
    reactions_count: Mapped[int] = mapped_column(Integer, nullable=True)
    
    """
    The last date the message was edited. Can be hidden.
    """
    last_edit_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    """
    The number of views the message has
    """
    views_count: Mapped[int] = mapped_column(Integer, nullable=True)

    """
    The number of times the message has been forwarded
    """
    forwards_count: Mapped[int] = mapped_column(Integer, nullable=True)
    
    """
    The number of replies to the message (Info about post comments (for channels) or message replies (for groups))
    """
    replies_count: Mapped[int] = mapped_column(Integer, nullable=True)    

    """
    The message entity this version belongs to
    """
    message: Mapped["Message"] = relationship(back_populates="versions")
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"))
    

    webpage: Mapped["Webpage"] = relationship('Webpage', back_populates='message_versions')
    webpage_id: Mapped[int] = mapped_column(ForeignKey("webpages.id"), nullable=True)
    """
    Webpage referenced in this message
    """    

    
    
class Actions(Base):
    __tablename__ = "actions"
    
    """
    id of the message of this action in the Telegram API
    """
    message_external_id: Mapped[int] = mapped_column(Integer, nullable=True)

    """
    The date the action was performed
    """
    date: Mapped[datetime] = mapped_column(DateTime)

    """
    The type of action that was performed
    """
    type: Mapped[str] = mapped_column(String)

    """
    The chat the action was performed in
    """
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id"))
    chat: Mapped["Chat"] = relationship("Chat", back_populates="actions")
    
    # enable Single Table Inheritance, so that all subclasses are stored in the same table. 
    # each subclass of Actions automatically gets a unique identifier based on the type attribute
    
    __mapper_args__ = {
        'polymorphic_identity': 'actions',
        'polymorphic_on': type
    }    

    
class PinMessage(Actions):
    __mapper_args__ = {'polymorphic_identity': 'PinMessage', 
                       'polymorphic_on': Actions.type}
    
class ChatCreate(Actions):
    """Group created"""
    __mapper_args__ = {'polymorphic_identity': 'ChatCreate',
                       'polymorphic_on': Actions.type}    
    title = mapped_column(String, nullable=True, use_existing_column=True)
    users = mapped_column(type_=JSON, nullable=True)
    
class EditTitle(Actions):
    """Group name changed"""
    __mapper_args__ = {'polymorphic_identity': 'EditTitle',
                       'polymorphic_on': Actions.type}
    title = mapped_column(String, nullable=True, use_existing_column=True)

class EditPhoto(Actions):
    """Group profile changed"""
    __mapper_args__ = {'polymorphic_identity': 'EditPhoto',
                       'polymorphic_on': Actions.type}
    photo_id = mapped_column(Integer, nullable=True)
    photo_access_hash = mapped_column(String, nullable=True)
    
class DeletePhoto(Actions):
    """Group profile deleted"""
    __mapper_args__ = {'polymorphic_identity': 'DeletePhoto',
                       'polymorphic_on': Actions.type}
    
class AddUser(Actions):
    """New member in the group"""
    __mapper_args__ = {'polymorphic_identity': 'AddUser',
                       'polymorphic_on': Actions.type}
    user_ids = mapped_column(type_=JSON, nullable=True)

class DeleteUser(Actions):
    """Member removed from the group"""
    __mapper_args__ = {'polymorphic_identity': 'DeleteUser',
                       'polymorphic_on': Actions.type}
    user_id = mapped_column(Integer, nullable=True)
    
class JoinedByLink(Actions):
    """Joined group by link"""
    __mapper_args__ = {'polymorphic_identity': 'JoinByLink',
                       'polymorphic_on': Actions.type}
    inviter_id = mapped_column(Integer, nullable=True)
    
class ChannelCreate(Actions):
    """Channel created"""
    __mapper_args__ = {'polymorphic_identity': 'ChannelCreate',
                       'polymorphic_on': Actions.type}
    original_channel_title = mapped_column(String, nullable=True) # title of the channel before it was converted to a group
    
class ChatMigrateTo(Actions):
    """Group converted to supergroup"""
    __mapper_args__ = {'polymorphic_identity': 'ChatMigrateTo',
                       'polymorphic_on': Actions.type}
    new_supergroup_id = mapped_column(Integer, nullable=True) # The supergroup it was migrated to
    
class ChannelMigrateFrom(Actions):
    """Indicates the channel was migrated from the specified chat"""
    __mapper_args__ = {'polymorphic_identity': 'ChannelMigrateFrom',
                       'polymorphic_on': Actions.type}
    original_title = mapped_column(String, nullable=True) # title of the channel before it was converted to a group
    original_chat_id = mapped_column(Integer, nullable=True) # The chat it was migrated from

class HistoryClear(Actions):
    """History cleared"""
    __mapper_args__ = {'polymorphic_identity': 'HistoryClear',
                       'polymorphic_on': Actions.type}
    
class GameScore(Actions):
    """Someone scored in a game"""
    __mapper_args__ = {'polymorphic_identity': 'GameScore',
                       'polymorphic_on': Actions.type}
    game_id = mapped_column(Integer, nullable=True)
    score = mapped_column(Integer, nullable=True)
    
# Some actions have been omitted, as they are not relevant for the current use case.
# For a full list of actions, see https://core.telegram.org/type/MessageAction

class Webpage(Base):
    """
    Represents a webpage in a message
    """
    
    __tablename__ = "webpages"
    
    external_id = mapped_column(Integer)
    """
    Preview ID of the webpage
    """
    
    url = mapped_column(String, nullable=True)
    """
    URL of previewed webpage
    """
    
    display_url = mapped_column(String, nullable=True)
    """
    Display URL of the webpage
    """
    
    type = mapped_column(String, nullable=True)
    """
    Type of the webpage. Can be: article, photo, audio, video, document, profile, app, or something else
    """
    
    site_name = mapped_column(String, nullable=True)
    """
    Short name of the site (e.g., Google Docs, App Store)
    """
    
    title = mapped_column(String, nullable=True)
    """
    Title of the content
    """
    
    description = mapped_column(String, nullable=True)
    """
    Description of the content
    """
    
    author = mapped_column(String, nullable=True)
    """
    author of the content
    """
    
    embed_url = mapped_column(String, nullable=True)
    """
    URL to show in the embedded preview
    """
    
    embed_type = mapped_column(String, nullable=True)
    """
    MIME type of the embedded preview, (e.g., text/html or video/mp4)
    """
    
    # Other attributes have been omitted, as they are not relevant for the current use case.
    # A full list of attributes can be found at https://core.telegram.org/constructor/webPage
    
    message_versions: Mapped[List["MessageVersion"]] = relationship(back_populates="webpage")
    """
    The message version this webpage was sent in
    """