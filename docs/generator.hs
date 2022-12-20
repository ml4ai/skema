{-# LANGUAGE OverloadedStrings #-}

module Main where

import Hakyll
import Text.Pandoc (
      WriterOptions
    , writerTemplate
    , writerTopLevelDivision
    , writerTableOfContents
    , writerNumberSections
    , writerHTMLMathMethod
    , HTMLMathMethod(MathJax)
    , compileTemplate
    , runPure
    , runWithDefaultPartials
    )
import Control.Monad (forM_)
import Data.Monoid (mappend)
import qualified Data.Map as M
import Data.Maybe (isJust, fromMaybe)
import System.Process (callCommand)

------------
-- Contexts
------------
postCtx :: Context String
postCtx =
  dateField "date" "%B %e, %Y"
  `mappend` defaultContext

archiveCtx posts =
  listField "posts" postCtx (return posts)
  `mappend` constField "title" "Posts"
  `mappend` defaultContext

------------
-- Options
------------

tocTemplate =
    either error id $ either (error . show) id $
    runPure $ runWithDefaultPartials $
    compileTemplate "" "<h2>Table of Contents</h2>$toc$\n$body$"

withTOC :: WriterOptions
withTOC = defaultHakyllWriterOptions{
    writerNumberSections  = True,
    writerTableOfContents = True,
    writerTemplate = Just tocTemplate,
    writerHTMLMathMethod = MathJax ""
}


withoutTOC :: WriterOptions
withoutTOC = defaultHakyllWriterOptions{
    writerHTMLMathMethod = MathJax ""
}

-------------
-- Compilers
-------------

compiler :: Compiler (Item String)
compiler = do
    csl <- load "apa.csl"
    biblio <- load "refs.bib"
    ident <- getUnderlying
    toc   <- getMetadataField ident "toc"
    let writerOptions' = case toc of
         Just _ ->  withTOC
         Nothing -> withoutTOC
    getResourceBody
        >>= readPandocBiblio defaultHakyllReaderOptions csl biblio
        >>= return . writePandocWith writerOptions'

------------
-- Rules
------------
templates :: Rules ()
templates = match "templates/*" $ compile templateCompiler

posts :: Rules ()
posts = match ("**.md" .&&. complement "README.md" .&&. complement "**index.md") $ do
    route $ setExtension "html"
    compile $ compiler
      >>= loadAndApplyTemplate "templates/post.html" defaultContext
      >>= relativizeUrls

archive :: Rules ()
archive = create ["posts.html"] $ do
    route $ setExtension "html"
    compile $ do
      posts <- recentFirst =<< loadAll "posts/*"
      makeItem ""
        >>= loadAndApplyTemplate "templates/list.html" (archiveCtx posts)
        >>= loadAndApplyTemplate "templates/index.html" (archiveCtx posts)
        >>= relativizeUrls

indices :: Rules ()
indices = match "**index.md" $ do
     route $ setExtension "html"
     compile $ compiler
      >>= loadAndApplyTemplate "templates/index.html" defaultContext
      >>= relativizeUrls

static :: Rules ()
static = forM_ ["fonts/*", "assets/**", "css/*", "js/*"] $ \x -> match x $ do
    route idRoute
    compile $ copyFileCompiler


------------
-- Main
------------
main :: IO ()
main = hakyllWith cfg $ do
  match "apa.csl" $ compile cslCompiler
  match "refs.bib"    $ compile biblioCompiler
  static
  indices
  posts
  templates
  archive
