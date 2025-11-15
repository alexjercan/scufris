import io
import wave
from piper import PiperVoice
from faster_whisper import WhisperModel
from copy import deepcopy
import asyncio
import os

import ollama
import discord
import dotenv

from scufris.common import setup_logger


def ragebaiter() -> None:
    logger = setup_logger()

    logger.debug("ragebaiter: loading environment variables")
    dotenv.load_dotenv()
    RAGEBAITER_DISCORD_TOKEN = os.environ["RAGEBAITER_DISCORD_TOKEN"]
    TESTING_GUILD_ID = int(os.getenv("TESTING_GUILD_ID", "1412415259683196970"))
    # RAGEBAITER_LANGUAGE = os.getenv("RAGEBAITER_LANGUAGE", "en")

    logger.debug("ragebaiter: loading opus library")
    discord.opus.load_opus("libopus.so")
    if not discord.opus.is_loaded():
        logger.critical("Could not load opus library")
        exit(1)

    logger.debug("ragebaiter: setting up discord bot")
    intents = discord.Intents.default()
    intents.voice_states = True
    bot = discord.Bot(intents=intents)

    model = WhisperModel(
        "distil-large-v3",
        device="cpu",
        compute_type="int8",
    )

    voice = PiperVoice.load("en_US-lessac-medium.onnx")

    connections = {}

    @bot.event
    async def on_ready():
        logger.debug(f"ragebaiter: logged in as {bot.user} (ID: {bot.user.id})")
        logger.info("ragebaiter: bot is ready")
        for vc in bot.voice_clients:
            await vc.disconnect(force=True)

    @bot.slash_command(guild_ids=[TESTING_GUILD_ID])
    async def hello(ctx: discord.ApplicationContext):
        logger.debug("ragebaiter: received /hello command")
        await ctx.respond("Hello!")

    @bot.slash_command(guild_ids=[TESTING_GUILD_ID])
    async def record(ctx: discord.ApplicationContext):
        logger.debug("ragebaiter: received /record command")
        await ctx.defer()

        logger.debug("ragebaiter: checking if user is in a voice channel")
        if not ctx.author.voice:
            logger.debug("ragebaiter: user is not in a voice channel")
            await ctx.respond("❌ You need to be in a voice channel to use this.")
            return

        logger.debug("ragebaiter: connecting to voice channel")
        if ctx.voice_client is None:
            logger.debug("ragebaiter: bot is not in a voice channel, connecting")
            vc = await ctx.author.voice.channel.connect()
        else:
            logger.debug("ragebaiter: bot is already in a voice channel")
            vc = ctx.voice_client

        logger.debug("ragebaiter: starting recording")
        connections[ctx.guild.id] = vc
        await ctx.respond("🎙️ Recording started! I'm listening...")

        logger.debug(f"ragebaiter: started recording task for guild {ctx.guild.id}")
        try:
            sink = discord.sinks.WaveSink()
            vc.start_recording(sink, record_audio_callback, ctx.channel, vc)
        except asyncio.CancelledError:
            logger.debug(
                f"ragebaiter: recording task for guild {ctx.guild.id} was cancelled"
            )
            vc.stop_recording()
        except Exception as e:
            logger.error(
                f"ragebaiter: error in recording task for guild {ctx.guild.id}: {e}"
            )

    async def record_audio_callback(
        sink: discord.sinks.WaveSink, channel: discord.TextChannel, vc: discord.VoiceClient
    ):
        logger.debug("ragebaiter: recording finished, preparing audio files")
        recorded_users = [f"<@{user_id}>" for user_id, _ in sink.audio_data.items()]
        logger.debug(f"ragebaiter: recorded users: {recorded_users}")

        # This is for debugging
        files = [
            discord.File(deepcopy(audio.file), f"{user_id}.{sink.encoding}")
            for user_id, audio in sink.audio_data.items()
        ]

        transcripts = []
        for user_id, audio in sink.audio_data.items():
            segments, info = model.transcribe(audio.file)

            buffers = []
            for _, segment in enumerate(segments):
                buffers.append(segment.text.strip())

            buffer = " ".join(buffers)
            transcripts.append(f"<@{user_id}>: {buffer}")

        transcript = "\n".join(transcripts)
        logger.debug(f"ragebaiter: transcript {transcript}")

        # This is for debugging
        logger.debug(f"ragebaiter: sending recorded {len(files)} audio files to channel")
        await channel.send(
            f"Finished recording audio for: {', '.join(recorded_users)}.\nWith transcript:\n{transcript}",
            files=files,
        )
        logger.debug("ragebaiter: audio files sent to channel")

        logger.debug("ragebaiter: generating LLM response")
        response = ollama.generate(
            model="gemma3:latest",
            prompt=transcript,
            system="",
        ).response
        await channel.send(response)
        logger.debug(f"ragebaiter: response {response}")

        logger.debug("ragebaiter: generating audio")
        with wave.open("test.wav", "wb") as wav_file:
            voice.synthesize_wav(response, wav_file)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize_wav(response, wav_file)

        wav_buffer.seek(0)
        files = [discord.File(wav_buffer, "response.wav")]
        await channel.send("Generated audio response", files=files)

        wav_buffer.seek(0)
        source = discord.FFmpegPCMAudio(wav_buffer, pipe=True)
        source = discord.PCMVolumeTransformer(source)
        vc.play(source)

    @bot.slash_command(guild_ids=[TESTING_GUILD_ID])
    async def stop(ctx: discord.ApplicationContext):
        logger.debug("ragebaiter: received /stop command")
        await ctx.defer()

        logger.debug("ragebaiter: disconnecting from voice channel")
        if ctx.guild.id in connections:
            logger.debug("ragebaiter: disconnected from voice channel")
            await ctx.respond("🛑 Record stopped and disconnected from voice channel.")

            vc = connections[ctx.guild.id]
            vc.stop_recording()
            del connections[ctx.guild.id]
            await ctx.delete()
            logger.debug("ragebaiter: cleaned up connection")
        else:
            logger.debug("ragebaiter: bot is not connected to a voice channel")
            await ctx.respond("❌ I'm not connected to a voice channel.")

    logger.info("ragebaiter: starting discord bot")
    bot.run(RAGEBAITER_DISCORD_TOKEN)
